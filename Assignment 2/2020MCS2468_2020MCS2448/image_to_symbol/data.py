import pandas as pd
import os
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import dgl
from copy import copy
import cv2

from classifier.test import classify
from classifier.model import get_classifier

def _basic_sudoku_graph():
    grids = [[0, 1, 2, 3, 8, 9, 10, 11],
             [4, 5, 6, 7, 12, 13, 14, 15],
             [16, 17, 18, 19, 24, 25, 26, 27],
             [20, 21, 22, 23, 28, 29, 30, 31],
             [32, 33, 34, 35, 40, 41, 42, 43],
             [36, 37, 38, 39, 44, 45, 46, 47],
             [48, 49, 50, 51, 56, 57, 58, 59],
             [52, 53, 54, 55, 60, 61, 62, 63]]

    edges = set()
    for i in range(64):
        row, col = i // 8, i % 8
        # same row and col
        row_src = row * 8
        col_src = col
        for _ in range(8):
            edges.add((row_src, i))
            edges.add((col_src, i))
            row_src += 1
            col_src += 8
        # same grid
        grid_row, grid_col = row // 2, col // 4

        for n in grids[grid_row * 2 + grid_col]:
            if n != i:
                edges.add((n, i))
    edges = list(edges)
    g = dgl.graph(edges)
    return g


class ImageToSymbolicDataset(Dataset):
    def __init__(self, classifier, query_path, target_path):
        self.classifier = classifier
        self.query_sudokus, self.target_sudokus = [], []
        for filename in os.listdir(query_path):
            self.query_sudokus.append(cv2.imread(os.path.join(query_path, filename)))
        for filename in os.listdir(query_path):
            self.target_sudokus.append(cv2.imread(os.path.join(target_path, filename)))

    def __len__(self):
        return len(self.query_sudokus)

    def __getitem__(self, item):
        image = self.query_sudokus[item]
        M, N = 28, 28
        digits = [image[x:x + M, y:y + N] for x in range(0, image.shape[0], M) for y in range(0, image.shape[1], N)]
        digit_tensor = torch.empty(size=(64, 1, 28, 28))
        for i in range(64):
            digit_tensor[i] = torch.tensor(digits[i].reshape(1, 28, 28))

        image = self.target_sudokus[item]
        target_symbolic = classify(self.classifier, image, is_target=True)
        return digit_tensor, target_symbolic


def sudoku_dataloader2(batch_size):
    dataset = ImageToSymbolicDataset('/content/image_dataset/train/target/',
                                     '/content/drive/MyDrive/Assignment2/10000_given_sudoku_dataset_csv/sudoku.csv')

    total_len = len(dataset)

    print(total_len)
    # data_sampler = RandomSampler(dataset)
    trainset, valset, testset = random_split(dataset, [int(total_len * 0.8),
                                                       int(total_len * 0.1),
                                                       int(total_len * 0.1)])

    basic_graph = _basic_sudoku_graph()

    def collate_fn(batch):
        batch_size = len(batch)
        graph_list = []
        q_batch = torch.empty(size=(64 * batch_size, 1, 28, 28))
        idx = 0
        for q, a in batch:
            q = torch.tensor(q, dtype=torch.long)
            a = torch.tensor(a, dtype=torch.long)
            graph = copy(basic_graph)
            graph.ndata['a'] = a  # a means answer
            graph_list.append(graph)
            q_batch[idx:idx + 64, :, :, :] = q
            idx += 64
        batch_graph = dgl.batch(graph_list)
        return batch_graph, q_batch

    trainloader = DataLoader(trainset, batch_size, collate_fn=collate_fn, drop_last=True)
    testloader = DataLoader(testset, batch_size, collate_fn=collate_fn, drop_last=True)
    valloader = DataLoader(valset, batch_size, collate_fn=collate_fn, drop_last=True)
    return trainloader, testloader, valloader
