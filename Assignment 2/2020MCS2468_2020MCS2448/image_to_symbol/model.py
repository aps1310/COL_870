import torch
from torch import nn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm
import numpy as np


class RRNLayer(nn.Module):
    def __init__(self, msg_layer, node_update_func, edge_drop):
        super(RRNLayer, self).__init__()
        self.msg_layer = msg_layer
        self.node_update_func = node_update_func
        self.edge_dropout = nn.Dropout(edge_drop)

    def forward(self, g):
        g.apply_edges(self.get_msg)
        g.edata['e'] = self.edge_dropout(g.edata['e'])
        g.update_all(message_func=fn.copy_e('e', 'msg'),
                     reduce_func=fn.sum('msg', 'm'))
        g.apply_nodes(self.node_update)

    def get_msg(self, edges):
        e = torch.cat([edges.src['h'], edges.dst['h']], -1)
        e = self.msg_layer(e)
        return {'e': e}

    def node_update(self, nodes):
        return self.node_update_func(nodes)


class RRN(nn.Module):
    def __init__(self,
                 msg_layer,
                 node_update_func,
                 num_steps,
                 edge_drop):
        super(RRN, self).__init__()
        self.num_steps = num_steps
        self.rrn_layer = RRNLayer(msg_layer, node_update_func, edge_drop)

    def forward(self, g, get_all_outputs=False):
        outputs = []
        for _ in range(self.num_steps):
            self.rrn_layer(g)
            if get_all_outputs:
                outputs.append(g.ndata['h'])
        if get_all_outputs:
            outputs = torch.stack(outputs, 0)  # num_steps x n_nodes x h_dim
        else:
            outputs = g.ndata['h']  # n_nodes x h_dim
        return outputs


class CNN_Sudoku(nn.Module):
    def __init__(self,
                 num_steps,
                 embed_size=16,
                 hidden_dim=84,
                 edge_drop=0.1):
        super(CNN_Sudoku, self).__init__()
        self.num_steps = num_steps

        self.digit_embed = nn.Embedding(9, embed_size)

        self.input_layer = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lstm = nn.LSTMCell(hidden_dim * 2, hidden_dim, bias=False)

        msg_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        ## CNN Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, embed_size)

        self.rrn = RRN(msg_layer, self.node_update_func, num_steps, edge_drop)

        self.output_layer = nn.Linear(hidden_dim, 9)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, g, q_batch, is_training=True):
        batch_size = q_batch.size(0) // 64
        labels = g.ndata.pop('a')

        ## CNN code
        cnnInput = F.relu(self.conv1(q_batch))
        cnnInput = F.relu(F.max_pool2d(self.conv2(cnnInput), 2))
        cnnInput = F.dropout(cnnInput, p=0.5, training=self.training)
        cnnInput = F.relu(F.max_pool2d(self.conv3(cnnInput), 2))
        cnnInput = F.dropout(cnnInput, p=0.5, training=self.training)
        cnnInput = cnnInput.view(-1, 3 * 3 * 64)
        cnnInput = F.relu(self.fc1(cnnInput))
        self.features = cnnInput
        cnnInput = F.dropout(cnnInput, training=self.training)
        cnnOutput = self.fc2(cnnInput)

        x = self.input_layer(torch.cat([cnnOutput], -1))

        g.ndata['x'] = x
        g.ndata['h'] = x
        g.ndata['rnn_h'] = torch.zeros_like(x, dtype=torch.float)
        g.ndata['rnn_c'] = torch.zeros_like(x, dtype=torch.float)

        outputs = self.rrn(g, is_training)
        logits = self.output_layer(outputs)

        preds = torch.argmax(logits, -1)

        if is_training:
            labels = torch.stack([labels] * self.num_steps, 0)
        logits = logits.view([-1, 9])
        labels = labels.view([-1])
        loss = self.loss_func(logits, labels)
        return preds, loss

    def node_update_func(self, nodes):
        x, h, m, c = nodes.data['x'], nodes.data['rnn_h'], nodes.data['m'], nodes.data['rnn_c']
        new_h, new_c = self.lstm(torch.cat([x, m], -1), (h, c))
        return {'h': new_h, 'rnn_c': new_c, 'rnn_h': new_h}

