import os
import cv2
import numpy as np

from classifier.model import get_classifier
from sudoku_solver.dataset import sudoku_dataloader
from sudoku_solver.test import solve_sudoku
from sudoku_solver.train import train
from sudoku_solver.model import SudokuNN
from sudoku_solver.utils import is_valid
from utils import get_default_device
import torch
import argparse
from classifier.test import classify
import pandas as pd


my_parser = argparse.ArgumentParser(allow_abbrev=False)

my_parser.add_argument('--train_dir', required=True, type=str, action='store')

my_parser.add_argument('--output_file', required=True, type=str, action='store', default='output.csv')

my_parser.add_argument('--test_dir', required=True, type=str, action='store')

my_parser.add_argument('--model_path', required=False, type=str, action='store')

args = my_parser.parse_args()

device = get_default_device()

classifier = get_classifier(device, 'classifier/classifier.pth')
classifier.eval()

trainloader, testloader, valloader = sudoku_dataloader(75, dataset_path=args.train_dir, classifier=classifier)

model = SudokuNN(num_steps=32, edge_drop=0.4)
model.to(device)
epoch = 100
if args.model_path:
    model.load_state_dict(torch.load(args.model_path))
else:
    train(model=model, train_loader=trainloader, val_loader=valloader, device=device, epoch=epoch)


sudoku_indices = np.arange(0, 64)
rows = sudoku_indices // 8
cols = sudoku_indices % 8


data = []
for filename in os.listdir(args.test_dir):
    query_image = cv2.imread(os.path.join(args.test_dir, filename), 0)
    symbolic_query = classify(classifier, query_image)
    pred = solve_sudoku(model=model, puzzle=symbolic_query, device=device)
    print(is_valid(pred))
    row = [filename]
    for symbol in pred:
        row.append(symbol)
    data.append(row)

output_df = pd.DataFrame(data)
output_df.to_csv(args.output_file, header=False, index=False)
