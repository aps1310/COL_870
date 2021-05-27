import os

import numpy as np
import pandas as pd
# import torch
# from torchvision.transforms import transforms
# from sudoku_solver.utils import is_valid
from classifier.model import get_classifier
# from utils import get_default_device
import cv2
import torch
from torch import optim
from torch.autograd import Variable
from torchvision.transforms import transforms
from tqdm import tqdm
import copy

from classifier.test import classify
from sudoku_solver.deterministic_solver.game import Sudoku

#
# classifier = get_classifier(get_default_device(), 'classifier/classifier.pth')
# classifier.eval()
#
# digit_count = np.zeros(9)
# num_of_images = 2000
# M, N = 28, 28
# query_data_folder = '../data/visual_sudoku/train/query/'
# target_data_folder = '../data/visual_sudoku/train/target/'
# digit_path = '../data/labelled_auto/'
#
#
# def auto_correct_target_image(model, target_image):
#     M, N = 28, 28
#     img = 255 - target_image
#     test_transform = transforms.Compose([transforms.ToPILImage(),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize(mean=[27.673422450965834], std=[72.5553937631818]),
#                                          ])
#     digits = [img[x:x + M, y:y + N] for x in range(0, img.shape[0], M) for y in range(0, img.shape[1], N)]
#     digits = np.array(digits)
#     # print(digits.shape)
#     new_digits = torch.zeros(digits.shape[0], 1, digits.shape[1], digits.shape[2])
#     # print(new_digits.shape)
#     for idx in range(len(digits)):
#         new_digits[idx, :, :, :] = test_transform(digits[idx]).reshape(1, 1, 28, 28)
#
#     pred = model(new_digits.cuda())
#     # print(pred.shape)
#     new_pred = pred[:, 1:10]
#     sudoku = torch.argmax(new_pred, dim=1) + 1
#
#     sudoku = sudoku.tolist()
#     # print(sudoku)
#     # print("Valid SUdoku: ",is_valid(sudoku)[0])
#     isValid, num = is_valid(sudoku)
#     tmp = copy.deepcopy(sudoku)
#
#     flag = True
#     if not isValid:
#         while not isValid:
#             sudoku = [0 if i == num else i for i in sudoku]
#             # print(sudoku)
#             isValid, num = is_valid(sudoku)
#
#         print("Not valid, Solving Now")
#         sudoku = np.array(sudoku).reshape((8, 8)).tolist()
#         if not Sudoku.is_empty(sudoku):
#             candidates_array = Sudoku.create_candidates_array(sudoku)
#             solved, empty_vals, methods, m_counts = Sudoku.solve_sudoku(candidates_array)
#             if empty_vals:
#                 print('Sudoku is not solvable')
#                 sudoku = tmp
#                 flag = False
#             elif solved:
#                 print('Sudoku solved')
#                 sudoku = np.array(candidates_array, dtype='uint8').flatten().tolist()
#             else:
#                 try:
#                     solved, dfs_depth, solution = Sudoku.solve_sudoku_dfs(candidates_array)
#                 except:
#                     solved = False
#                 if not solved:
#                     print('Sudoku is not solvable')
#                     sudoku = tmp
#                     flag = False
#                 else:
#                     print('Sudoku solved')
#                     sudoku = np.array(solution, dtype='uint8').flatten().tolist()
#
#         else:
#             print('You entered an empty Sudoku')
#             sudoku = tmp
#             flag = False
#
#     return sudoku, flag
#
#
# def compare_get_invalid_digit(queryDigits, targetDigits):
#     invalid_digits = []
#     for i in range(64):
#         if not (np.array_equal(queryDigits[i], targetDigits[i])):
#             invalid_digits.append(1)
#         else:
#             invalid_digits.append(0)
#
#     x = np.array(invalid_digits)
#     invalid_index = np.where(x == 1)[0].tolist()
#     return invalid_index
#
#
# for i in tqdm(range(num_of_images)):
#     queryImage = cv2.imread(query_data_folder + str(i) + '.png', 0)
#     targetImage = cv2.imread(target_data_folder + str(i) + '.png', 0)
#     queryDigits = [queryImage[x:x + M, y:y + N] for x in range(0, queryImage.shape[0], M) for y in
#                    range(0, queryImage.shape[1], N)]
#     targetDigits = [targetImage[x:x + M, y:y + N] for x in range(0, targetImage.shape[0], M) for y in
#                     range(0, targetImage.shape[1], N)]
#     tagetLabelList, was_solved = auto_correct_target_image(classifier, targetImage)
#     if not was_solved:
#         print("Couldn't be solved removed")
#         continue
#     query_invalid_digit_index = compare_get_invalid_digit(queryDigits, targetDigits)
#     # save Zero class digits
#
#     for j, idx in enumerate(query_invalid_digit_index):
#         cv2.imwrite(digit_path + '0/' + str(int(digit_count[0])) + '.png', 255 - queryDigits[idx])
#         digit_count[0] += 1
#     for k in range(64):
#         label = tagetLabelList[k]
#         cv2.imwrite(digit_path + str(label) + '/' + str(int(digit_count[label])) + '.png', 255 - targetDigits[k])
#         digit_count[label] += 1
# import csv
#
# row = 0
# writer = csv.writer(open('x.csv', 'w', newline=''), delimiter=',')
# writer2 = csv.writer(open('y.csv', 'w', newline=''), delimiter=',')
# for i in range(9):
#     digit_path = '../data/labelled_auto/{}/'.format(i)
#     for filename in tqdm(os.listdir(digit_path)):
#         img = cv2.imread(os.path.join(digit_path, filename), 0)
#         writer.writerow(img.flatten().tolist())
#         writer2.writerow([i])
#

# import csv
# from utils import get_default_device
# from classifier.model import get_classifier
#
# classifier = get_classifier(get_default_device(), 'classifier/best_classifier.pth')
#
#
# x_csv = open('../data/visual_sudoku/train/target.csv', 'w', newline='')
# writer = csv.writer(x_csv, delimiter=',')
#
# img_name = 0
# idx_to_label = {}
# label_to_idx = {x: [] for x in range(9)}
#
# num_of_labels = 1000
# # Generate the data
# for filename in os.listdir('../data/visual_sudoku/train/target'):
#     query_image = cv2.imread(os.path.join('../data/visual_sudoku/train/target', filename), 0)
#     symbolic_query = classify(classifier, query_image)
#     writer.writerow(symbolic_query)

# from torch.utils.data import Dataset, DataLoader
# import os
# from torchvision.transforms import transforms
#
#
# class VisualSudoku(Dataset):
#     def __init__(self, img_folder, solution_df):
#         self.img_folder = img_folder
#         self.solution_df = solution_df
#         self.dataset_size = len(self.solution_df)
#         assert sum([len(folder) for r, d, folder in os.walk(img_folder)]) == len(self.solution_df)
#
#     def __len__(self):
#         return self.dataset_size
#
#     def __getitem__(self, n):
#         img = cv2.imread(os.path.join(self.img_folder, str(n) + '.png'), 0)
#         digits = np.array([img[x:x + 28, y:y + 28].flatten() for x in range(0, img.shape[0], 28) for y in range(0, img.shape[1], 28)])
#         label = self.solution_df.iloc[n].values
#
#         return digits, label
#
#
# def to_device(data, device):
#     """Move tensor(s) to chosen device"""
#     if isinstance(data, (list, tuple)):
#         return [to_device(x, device) for x in data]
#     return data.to(device, non_blocking=True)
#
#
# class DeviceDataLoader:
#     """Wrap a dataloader to move data to a device"""
#
#     def __init__(self, dl, device):
#         self.dl = dl
#         self.device = device
#
#     def __iter__(self):
#         """Yield a batch of data after moving it to device"""
#         for b in self.dl:
#             yield to_device(b, self.device)
#
#     def __len__(self):
#         """Number of batches"""
#         return len(self.dl)
#
#
# def get_dataloader(device, csv_path, image_path, batch_size=64):
#     y = pd.read_csv(os.path.join(csv_path, 'target.csv'), header=None)
#     dataset = VisualSudoku(os.path.join(image_path), y)
#     trainset, testset = torch.utils.data.random_split(dataset, (8000, 2000))
#     num_workers = 0 if os.name == 'nt' else 2
#     train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     train_dataloader = DeviceDataLoader(train_dataloader, device)
#
#     test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     test_dataloader = DeviceDataLoader(test_dataloader, device)
#     return train_dataloader, test_dataloader
#
#
# def get_default_device():
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu')
#
#
# from collections import OrderedDict
#
# import torch.nn as nn
# import torch
# import torch.nn.functional as F
#
#
# def create_constraint_mask():
#     constraint_mask = torch.zeros((64, 3, 64), dtype=torch.float)
#     # row constraints
#     for a in range(64):
#         r = 8 * (a // 8)
#         for b in range(8):
#             constraint_mask[a, 0, r + b] = 1
#     # column constraints
#     for a in range(64):
#         c = a % 8
#         for b in range(8):
#             constraint_mask[a, 1, c + 8 * b] = 1
#     # box constraints
#     for a in range(64):
#         r = a // 8
#         c = a % 8
#         br = r // 2
#         bc = c // 4
#         for b in range(8):
#             r = b // 4
#             constraint_mask[a, 2, 16 * br + 4 * bc + b + r * 4] = 1
#     return constraint_mask
#
#
# class MLP(nn.Module):
#     def __init__(self, input_dims=784, n_hiddens=(256, 256), n_class=10):
#         super(MLP, self).__init__()
#         assert isinstance(input_dims, int), 'Please provide int for input_dims'
#         self.input_dims = input_dims
#         current_dims = input_dims
#         layers = OrderedDict()
#
#         if isinstance(n_hiddens, int):
#             n_hiddens = [n_hiddens]
#         else:
#             n_hiddens = list(n_hiddens)
#         for i, n_hidden in enumerate(n_hiddens):
#             layers['fc{}'.format(i + 1)] = nn.Linear(current_dims, n_hidden)
#             layers['relu{}'.format(i + 1)] = nn.ReLU()
#             layers['drop{}'.format(i + 1)] = nn.Dropout(0.2)
#             current_dims = n_hidden
#         layers['out'] = nn.Linear(current_dims, n_class)
#
#         self.model = nn.Sequential(layers)
#         print(self.model)
#
#     def forward(self, input):
#         input = input.view(input.size(0), -1)
#         assert input.size(1) == self.input_dims
#         return self.model.forward(input)
#
#
# class SudokuSolver(nn.Module):
#     def __init__(self, constraint_mask, n=8, hidden1=100):
#         super(SudokuSolver, self).__init__()
#         self.constraint_mask = constraint_mask.view(1, n * n, 3, n * n, 1)
#         self.n = n
#         self.hidden1 = hidden1
#
#         # Feature vector is the 3 constraints
#         self.input_size = 3 * n
#
#         self.l1 = nn.Linear(self.input_size,
#                             self.hidden1, bias=False)
#         self.a1 = nn.ReLU()
#         self.l2 = nn.Linear(self.hidden1,
#                             n, bias=False)
#         self.softmax = nn.Softmax(dim=1)
#
#     # x is a (batch, n^2, n) tensor
#     def forward(self, x):
#         n = self.n
#         bts = x.shape[0]
#         c = self.constraint_mask
#         min_empty = (x.sum(dim=2) == 0).sum(dim=1).max()
#         x_pred = x.clone()
#         for a in range(min_empty):
#             # score empty numbers
#             constraints = (x.view(bts, 1, 1, n * n, n) * c).sum(dim=3)
#             # empty cells
#             empty_mask = (x.sum(dim=2) == 0)
#
#             f = constraints.reshape(bts, n * n, 3 * n)
#             y_ = self.l2(self.a1(self.l1(f[empty_mask])))
#
#             s_ = self.softmax(y_)
#
#             # Score the rows
#             x_pred[empty_mask] = s_
#
#             s = torch.zeros_like(x_pred)
#             s[empty_mask] = s_
#             # find most probable guess
#             score, score_pos = s.max(dim=2)
#             mmax = score.max(dim=1)[1]
#             # fill it in
#             nz = empty_mask.sum(dim=1).nonzero().view(-1)
#             mmax_ = mmax[nz]
#             ones = torch.ones(nz.shape[0])
#             x.index_put_((nz, mmax_, score_pos[nz, mmax_]), ones)
#         return x_pred, x
#
#
# class VisualSolver(nn.Module):
#     def __init__(self):
#         super(VisualSolver, self).__init__()
#         self.classifier = MLP()
#         self.solver = SudokuSolver(create_constraint_mask())
#
#     def forward(self, x):
#         labels = self.classifier(x)
#         solution = self.solver(labels)
#
#
# def get_visual_sudoku_solver(device):
#     model = VisualSolver()
#     model.to(device)
#     return model
#
#
#
#
# device = get_default_device()
# model = get_visual_sudoku_solver(device)
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
# trainloader, testloader = get_dataloader(device, '../data/visual_sudoku/train/', '../data/visual_sudoku/train/target')
#
#
#
# for epoch in range(10):
#         model.train()
#         for batch_idx, (data, target) in enumerate(trainloader):
#             indx_target = target.clone()
#             data, target = Variable(data.cuda()), Variable(target.cuda())
#
#             optimizer.zero_grad()
#             output = model(data)
#             loss = F.cross_entropy(output, target)
#             loss.backward()
#             optimizer.step()

# from sudoku_solver.utils import is_valid
# from utils import get_default_device
# from tqdm import tqdm
#
# query_image_folder = '../data/visual_sudoku/train/query/'
# target_image_folder = '../data/visual_sudoku/train/target/'
# transform = transforms.Compose([
#                         transforms.ToPILImage(),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean=[27.67], std=[72.55]),
#                      ])
# model = get_classifier(get_default_device(), 'classifier/classifier.pth')
#
# with open('sudoku.csv', 'w') as sudoku_csv:
#     sudoku_csv.write('quizzes, solutions\n')
#     for idx in tqdm(range(10000)):
#         query_image = cv2.imread(query_image_folder+str(idx)+'.png', 0)
#         target_image = cv2.imread(target_image_folder+str(idx)+'.png', 0)
#         # if transform:
#         #     query_image = transform(query_image)
#         #     target_image = transform(target_image)
#
#         M, N = 28, 28
#         digits_q = [query_image[x:x + M, y:y + N] for x in range(0, query_image.shape[0], M) for y in range(0, query_image.shape[1], N)]
#         digits_t = [target_image[x:x + M, y:y + N] for x in range(0, target_image.shape[0], M) for y in range(0, target_image.shape[1], N)]
#
#         target_symbolic = classify(model, target_image)
#
#         # Use some deterministic logic to validate the classifier, if the output is incorrect try to correct
#         isValid, num = is_valid(target_symbolic)
#         tmp = copy.deepcopy(target_symbolic)
#         if not isValid:
#             while not isValid:
#                 target_symbolic = [0 if i == num else i for i in target_symbolic]
#                 isValid, num = is_valid(target_symbolic)
#
#             target_symbolic = np.array(target_symbolic).reshape((8, 8)).tolist()
#             if not Sudoku.is_empty(target_symbolic):
#                 candidates_array = Sudoku.create_candidates_array(target_symbolic)
#                 solved, empty_vals, methods, m_counts = Sudoku.solve_sudoku(candidates_array)
#                 if empty_vals:
#                     target_symbolic = tmp
#                 elif solved:
#                     target_symbolic = np.array(candidates_array, dtype='uint8').flatten().tolist()
#                 else:
#                     solved, dfs_depth, solution = Sudoku.solve_sudoku_dfs(candidates_array)
#                     if not solved:
#                         target_symbolic = tmp
#                     else:
#                         target_symbolic = np.array(solution, dtype='uint8').flatten().tolist()
#             else:
#                 target_symbolic = tmp
#
#         # Since query image is same as target except at the blanks, use this to correctly classify blanks
#         query_symbolic = [0]*64
#         for i in range(64):
#             if np.array_equal(digits_q[i], digits_t[i]):
#                 query_symbolic[i] = target_symbolic[i]
#
#         sudoku_csv.write(''.join(map(str, query_symbolic))+', '+''.join(map(str, target_symbolic))+'\n')
#

# c = 0
# dataset = pd.read_csv("sudoku.csv", sep=', ')
# for i in range(10000):
#     if is_valid(list(map(int, [char for char in dataset.quizzes.iloc[i]]))):
#         c += 1
#         print(c)

# import csv
# import os
# import urllib.request
# import zipfile
# import numpy as np
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
# from torch.utils.data.dataset import Dataset
# import dgl
# from copy import copy
#
#
# def _basic_sudoku_graph():
#     grids = [[0, 1, 2, 3, 8, 9, 10, 11],
#              [4, 5, 6, 7, 12, 13, 14, 15],
#              [16, 17, 18, 19, 24, 25, 26, 27],
#              [20, 21, 22, 23, 28, 29, 30, 31],
#              [32, 33, 34, 35, 40, 41, 42, 43],
#              [36, 37, 38, 39, 44, 45, 46, 47],
#              [48, 49, 50, 51, 56, 57, 58, 59],
#              [52, 53, 54, 55, 60, 61, 62, 63]]
#
#     edges = set()
#     for i in range(64):
#         row, col = i // 8, i % 8
#         # same row and col
#         row_src = row * 8
#         col_src = col
#         for _ in range(8):
#             edges.add((row_src, i))
#             edges.add((col_src, i))
#             row_src += 1
#             col_src += 8
#         # same grid
#         grid_row, grid_col = row // 2, col // 4
#
#         for n in grids[grid_row * 2 + grid_col]:
#             if n != i:
#                 edges.add((n, i))
#     edges = list(edges)
#     print(edges)
#
#     def get_edge(node, edges):
#         l = 0
#         for e in edges:
#             if node == e[0]:
#                 l += 1
#                 # print(e)
#         print(l)
#     for i in range(64):
#         get_edge(i, edges)
#     g = dgl.graph(edges)
#     return g
#
#
#
# print(_basic_sudoku_graph())
#
# sudoku_indices = np.arange(0, 64)
# rows = sudoku_indices // 8
# cols = sudoku_indices % 8
#
# print(rows, cols)


# from classifier.model import get_classifier
# from sudoku_solver.dataset import sudoku_dataloader
# from sudoku_solver.test import solve_sudoku
# from sudoku_solver.train import train
# from sudoku_solver.model import SudokuNN
# from sudoku_solver.utils import is_valid
# from utils import get_default_device
# import torch
#
# pretrained_model_path = 'sudoku_solver/best_model_383.pt'
# device = get_default_device()
# model = SudokuNN(num_steps=32, edge_drop=0.4)
# model.to(device)
#
# checkpoint = torch.load(pretrained_model_path)
# model.load_state_dict(checkpoint['state_dict'])
#
#
# sudoku_indices = np.arange(0, 64)
# rows = sudoku_indices // 8
# cols = sudoku_indices % 8
#
#
# df = pd.read_csv('med_sudoku.csv', header=None, index_col=None)
# correct = 0
# for i in tqdm(range(225)):
#     symbolic_query = list(map(int, list(list(df.iloc[i])[0])))
#     pred = solve_sudoku(model=model, puzzle=symbolic_query, device=device)
#     if is_valid(pred):
#         correct += 1
#
# print(correct)


class ImageToSymbolicDataset:
    def __init__(self, image_path, csv_path):
        self.image_path = image_path
        self.df = pd.read_csv(csv_path, sep=', ', index_col=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image = cv2.imread(os.path.join(self.image_path, '{}.png'.format(item)), 0)
        target_symbolic = list(map(int, list(self.df.iloc[item][1])))
        M, N = 28, 28
        digits = [image[x:x + M, y:y + N] for x in range(0, image.shape[0], M) for y in range(0, image.shape[1], N)]
        digit_tensor = torch.empty(size=(64, 1, 28, 28))
        for i in range(64):
            digit_tensor[i] = torch.tensor(digits[i].reshape(1, 28, 28))
        return digit_tensor, target_symbolic


dataset = ImageToSymbolicDataset('../data/visual_sudoku/train/query/', 'sudoku.csv')
print(dataset[0][0].shape)