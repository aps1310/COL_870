from torch import nn
import torch
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import torch.utils
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torch.nn import Parameter, RNNCellBase
import time


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, char_vocab_size, num_tags, embedding_dim=100, enable_char=False, char_embedding_dim=50,
                 char_lstm_dim=25, hidden_dim=100, num_layers=1, embFileName=None, dropout=0.5, clippingValue=5,
                 enable_layer_norm=False):
        super(BiLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.char_embedding_dim = char_embedding_dim
        self.enable_char = enable_char
        self.char_lstm_dim = char_lstm_dim
        self.enable_layer_norm = enable_layer_norm
        self.gradientClipingValue = clippingValue

        if embFileName is not None:
            glove2word2vec(glove_input_file=embFileName,
                           word2vec_output_file="ner/data/emb_word2vec_format.txt")
            mdl = gensim.models.KeyedVectors.load_word2vec_format("ner/data/emb_word2vec_format.txt")
            weights = torch.FloatTensor(mdl.vectors)
            self.word_embeds = nn.Embedding.from_pretrained(weights, padding_idx=num_tags)
        else:
            self.word_embeds = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=num_tags)

        self.drop = nn.Dropout(dropout)
        if enable_char:
            self.char_embeds = nn.Embedding(char_vocab_size+1, char_embedding_dim, padding_idx=char_vocab_size)
            self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True,
                                     batch_first=True)
            if self.enable_layer_norm:
                self.lstm_forward = LayerNormLSTMCell(embedding_dim + char_lstm_dim * 2, hidden_dim)
                self.lstm_backward = LayerNormLSTMCell(embedding_dim + char_lstm_dim * 2, hidden_dim)
            else:
                self.lstm = nn.LSTM(embedding_dim + char_lstm_dim * 2, hidden_dim,
                                    num_layers=1, bidirectional=True, batch_first=True)
        else:
            if self.enable_layer_norm:
                self.lstm_forward = LayerNormLSTMCell(embedding_dim, hidden_dim)
                self.lstm_backward = LayerNormLSTMCell(embedding_dim, hidden_dim)
            else:
                self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                                    num_layers=1, bidirectional=True, batch_first=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, sentence, chars=None, chars_length=None):
        word_embeds = self.word_embeds(sentence)
        max_sentence_len = sentence.shape[1]
        batch_size = sentence.shape[0]
        if chars is not None:
            max_word_len = chars.shape[2]
            char_embeds = self.char_embeds(chars)
            char_embeds = char_embeds.reshape((char_embeds.shape[0] * char_embeds.shape[1],
                                               char_embeds.shape[2], char_embeds.shape[3]))
            lstm_out, _ = self.char_lstm(char_embeds)

            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((lstm_out.size(0), lstm_out.size(2))))).to(sentence.device)
            for i in range(lstm_out.shape[0]):
                chars_embeds_temp[i] = torch.cat(
                    (lstm_out[i, max_word_len - 1, :self.char_lstm_dim], lstm_out[i, 0, self.char_lstm_dim:]))
            char_lstm = chars_embeds_temp.reshape(batch_size, max_sentence_len, chars_embeds_temp.shape[1])
            word_embeds = torch.cat([word_embeds, char_lstm], 2)

        word_embeds = word_embeds.reshape((word_embeds.shape[0] * word_embeds.shape[1],
                                           word_embeds.shape[2]))

        word_embeds = self.drop(word_embeds)

        if self.enable_layer_norm:
            out_forward, _ = self.lstm_forward(word_embeds, None)
            out_backward, _ = self.lstm_backward(torch.flip(word_embeds, [0]), None)
            out = torch.cat((out_forward, out_backward), 1)
            out = out.reshape((batch_size, max_sentence_len,
                               out.shape[1]))
        else:
            word_embeds = word_embeds.reshape((batch_size,
                                               word_embeds.shape[0] // batch_size,
                                               word_embeds.shape[1]))
            out, _ = self.lstm(word_embeds, None)

        out = out.reshape(-1, out.shape[2])
        tag_space = self.hidden2tag(out)
        F.log_softmax(tag_space, dim=1)
        return tag_space


class LayerNormLSTMCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormLSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

        self.ln_ingate = nn.LayerNorm(hidden_size)
        self.ln_forgetgate = nn.LayerNorm(hidden_size)
        self.ln_cellgate = nn.LayerNorm(hidden_size)
        self.ln_outgate = nn.LayerNorm(hidden_size)
        self.ln_cy = nn.LayerNorm(hidden_size)
        self.ln = {
            'ingate': self.ln_ingate,
            'forgetgate': self.ln_forgetgate,
            'cellgate': self.ln_cellgate,
            'outgate': self.ln_outgate,
            'cy': self.ln_cy
        }

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')

        hx, cx = hx
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # use layer norm here
        ingate = torch.sigmoid(self.ln['ingate'](ingate))
        forgetgate = torch.sigmoid(self.ln['forgetgate'](forgetgate))
        cellgate = torch.tanh(self.ln['cellgate'](cellgate))
        outgate = torch.sigmoid(self.ln['outgate'](outgate))

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(self.ln['cy'](cy))

        return hy, cy


# ############################################# BILSTM-CRF #############################################################

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, char_vocab_size, num_tags, embedding_dim=100, enable_char=False, char_embedding_dim=50,
                 char_lstm_dim=25, hidden_dim=100, num_layers=1, embFileName=None, dropout=0.5, clippingValue=5,
                 enable_layer_norm=False):
        super(BiLSTM_CRF, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.char_embedding_dim = char_embedding_dim
        self.enable_char = enable_char
        self.char_lstm_dim = char_lstm_dim
        self.enable_layer_norm = enable_layer_norm
        self.gradientClipingValue = clippingValue

        if embFileName is not None:
            glove2word2vec(glove_input_file=embFileName,
                           word2vec_output_file="ner/data/emb_word2vec_format.txt")
            mdl = gensim.models.KeyedVectors.load_word2vec_format("ner/data/emb_word2vec_format.txt")
            weights = torch.FloatTensor(mdl.vectors)
            self.word_embeds = nn.Embedding.from_pretrained(weights, padding_idx=num_tags)
        else:
            self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=num_tags)

        self.drop = nn.Dropout(dropout)
        if enable_char:
            self.char_embeds = nn.Embedding(char_vocab_size + 1, char_embedding_dim, padding_idx=char_vocab_size)
            self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True,
                                     batch_first=True)
            if self.enable_layer_norm:
                self.lstm_forward = LayerNormLSTMCell(embedding_dim + char_lstm_dim * 2, hidden_dim)
                self.lstm_backward = LayerNormLSTMCell(embedding_dim + char_lstm_dim * 2, hidden_dim)
            else:
                self.lstm = nn.LSTM(embedding_dim + char_lstm_dim * 2, hidden_dim,
                                    num_layers=1, bidirectional=True, batch_first=True)
        else:
            if self.enable_layer_norm:
                self.lstm_forward = LayerNormLSTMCell(embedding_dim, hidden_dim)
                self.lstm_backward = LayerNormLSTMCell(embedding_dim, hidden_dim)
            else:
                self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                                    num_layers=1, bidirectional=True, batch_first=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_tags)

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)

        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward_algorithm(self, emissions, mask):
        seq_length = emissions.size(0)

        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)

            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions

            next_score = torch.logsumexp(next_score, dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def _get_lstm_features(self, sentence, chars=None, chars_length=None):
        word_embeds = self.word_embeds(sentence)
        max_sentence_len = sentence.shape[1]
        batch_size = sentence.shape[0]
        if chars is not None:
            max_word_len = chars.shape[2]
            char_embeds = self.char_embeds(chars)
            char_embeds = char_embeds.reshape((char_embeds.shape[0] * char_embeds.shape[1],
                                               char_embeds.shape[2], char_embeds.shape[3]))
            lstm_out, _ = self.char_lstm(char_embeds)

            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((lstm_out.size(0), lstm_out.size(2))))).to(
                sentence.device)
            for i in range(lstm_out.shape[0]):
                chars_embeds_temp[i] = torch.cat(
                    (lstm_out[i, max_word_len - 1, :self.char_lstm_dim], lstm_out[i, 0, self.char_lstm_dim:]))
            char_lstm = chars_embeds_temp.reshape(batch_size, max_sentence_len, chars_embeds_temp.shape[1])
            word_embeds = torch.cat([word_embeds, char_lstm], 2)

        word_embeds = word_embeds.reshape((word_embeds.shape[0] * word_embeds.shape[1],
                                           word_embeds.shape[2]))

        word_embeds = self.drop(word_embeds)

        if self.enable_layer_norm:
            out_forward, _ = self.lstm_forward(word_embeds, None)
            out_backward, _ = self.lstm_backward(torch.flip(word_embeds, [0]), None)
            out = torch.cat((out_forward, out_backward), 1)
            out = out.reshape((batch_size, max_sentence_len,
                               out.shape[1]))
        else:
            word_embeds = word_embeds.reshape((batch_size,
                                               word_embeds.shape[0] // batch_size,
                                               word_embeds.shape[1]))
            out, _ = self.lstm(word_embeds, None)

        tag_space = self.hidden2tag(out)

        return tag_space

    def _score_sentence(self, emissions, tags, mask):
        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)
        tags[mask == False] = 0

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        seq_ends = mask.long().sum(dim=0) - 1

        last_tags = tags[seq_ends, torch.arange(batch_size)]

        score += self.end_transitions[last_tags]

        return score

    def _viterbi_decode(self, emissions, mask):
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)

            broadcast_emission = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emission

            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions

        seq_ends = mask.long().sum(dim=0)
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

    def neg_log_likelihood(self, sentence, tags, mask, chars=None, chars_length=None):
        feats = self._get_lstm_features(sentence, chars, chars_length)

        # The below algorithms assumes batch-second, hence reshaping
        feats = feats.transpose(0, 1)
        mask = mask.transpose(0, 1)
        tags = tags.transpose(0, 1)

        denominator = self.forward_algorithm(feats, mask)
        numerator = self._score_sentence(feats, tags, mask)
        return (denominator - numerator).sum()

    def forward(self, sentence, mask,  chars=None, chars_length=None):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, chars, chars_length)

        feats = lstm_feats.transpose(0, 1)
        mask = mask.transpose(0, 1)

        # Find the best path, given the features.
        tag_seq = self._viterbi_decode(feats, mask)
        return tag_seq

