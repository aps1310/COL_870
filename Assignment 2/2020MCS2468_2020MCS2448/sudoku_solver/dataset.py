import cv2
from torchvision.transforms import transforms

from classifier.test import classify
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
import dgl
from copy import deepcopy, copy
import torch
from sudoku_solver.utils import is_valid
from sudoku_solver.deterministic_solver.game import Sudoku


class SudokuDataset(Dataset):

    def __init__(self, data_path, classifier, transform=None):
        self.data_path = data_path
        self.query_image_folder = data_path+'visual_sudoku/train/query/'
        self.target_image_folder = data_path + 'visual_sudoku/train/target/'
        self.transform = transform
        self.classifier = classifier
        num_query_images = len(os.listdir(self.query_image_folder))
        num_target_images = len(os.listdir(self.target_image_folder))
        assert num_query_images == num_target_images
        self.dataset_size = num_query_images

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        query_image = cv2.imread(self.query_image_folder+str(idx)+'.png', 0)
        target_image = cv2.imread(self.target_image_folder+str(idx)+'.png', 0)
        # if self.transform:
        #     query_image = self.transform(query_image)
        #     target_image = self.transform(target_image)

        M, N = 28, 28
        digits_q = [query_image[x:x + M, y:y + N] for x in range(0, query_image.shape[0], M) for y in range(0, query_image.shape[1], N)]
        digit_t = [target_image[x:x + M, y:y + N] for x in range(0, target_image.shape[0], M) for y in range(0, target_image.shape[1], N)]

        target_symbolic = classify(self.classifier, target_image)

        # Use some deterministic logic to validate the classifier, if the output is incorrect try to correct
        isValid, num = is_valid(target_symbolic)
        tmp = deepcopy(target_symbolic)
        if not isValid:
            while not isValid:
                target_symbolic = [0 if i == num else i for i in target_symbolic]
                isValid, num = is_valid(target_symbolic)

            target_symbolic = np.array(target_symbolic).reshape((8, 8)).tolist()
            if not Sudoku.is_empty(target_symbolic):
                candidates_array = Sudoku.create_candidates_array(target_symbolic)
                solved, empty_vals, methods, m_counts = Sudoku.solve_sudoku(candidates_array)
                if empty_vals:
                    target_symbolic = tmp
                elif solved:
                    target_symbolic = np.array(candidates_array, dtype='uint8').flatten().tolist()
                else:
                    solved, dfs_depth, solution = Sudoku.solve_sudoku_dfs(candidates_array)
                    if not solved:
                        target_symbolic = tmp
                    else:
                        target_symbolic = np.array(solution, dtype='uint8').flatten().tolist()
            else:
                target_symbolic = tmp

        # Since query image is same as target except at the blanks, use this to correctly classify blanks
        query_symbolic = []
        for i in range(64):
            if not (np.array_equal(digits_q[i], digit_t[i])):
                query_symbolic.append(1)
            else:
                query_symbolic.append(0)

        for i in range(64):
            if query_symbolic[i] == 1:
                query_symbolic[i] = target_symbolic[i]

        return target_symbolic, query_symbolic


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


def sudoku_dataloader(batch_size, dataset_path, classifier):

    test_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[27.67], std=[72.55]),
                     ])
    dataset = SudokuDataset(dataset_path, transform=test_transform, classifier=classifier)

    trainset, valset, testset = random_split(dataset, [8000, 1000, 1000])

    basic_graph = _basic_sudoku_graph()
    sudoku_indices = np.arange(0, 64)
    rows = sudoku_indices // 8
    cols = sudoku_indices % 8

    def collate_fn(batch):
        graph_list = []
        for q, a in batch:
            q = torch.tensor(q, dtype=torch.long)
            a = torch.tensor(a, dtype=torch.long)
            graph = copy(basic_graph)
            graph.ndata['q'] = q  # q means question
            graph.ndata['a'] = a  # a means answer
            graph.ndata['row'] = torch.tensor(rows, dtype=torch.long)
            graph.ndata['col'] = torch.tensor(cols, dtype=torch.long)
            graph_list.append(graph)
        batch_graph = dgl.batch(graph_list)
        return batch_graph

    trainloader = DataLoader(trainset, batch_size, collate_fn=collate_fn, drop_last=True)
    testloader = DataLoader(testset, batch_size, collate_fn=collate_fn, drop_last=True)
    valloader = DataLoader(valset, batch_size, collate_fn=collate_fn, drop_last=True)
    return trainloader, testloader, valloader



basic_graph = _basic_sudoku_graph()
sudoku_indices = np.arange(0, 64)
rows = sudoku_indices // 8
cols = sudoku_indices % 8


def get_sudoku_graph(sudoku):
    graph_list = []
    q = torch.tensor(sudoku, dtype=torch.long)
    graph = copy(basic_graph)
    graph.ndata['q'] = q  # q means question
    graph.ndata['row'] = torch.tensor(rows, dtype=torch.long)
    graph.ndata['col'] = torch.tensor(cols, dtype=torch.long)
    graph_list.append(graph)
    batch_graph = dgl.batch(graph_list)
    return batch_graph
