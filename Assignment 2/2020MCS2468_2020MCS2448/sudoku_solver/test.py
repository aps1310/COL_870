import numpy as np
import torch

from sudoku_solver.dataset import _basic_sudoku_graph


def solve_sudoku(model, device, puzzle):

    puzzle = np.array(puzzle, dtype=np.long).reshape([-1])

    g = _basic_sudoku_graph().to(device)
    sudoku_indices = np.arange(0, 64)
    rows = sudoku_indices // 8
    cols = sudoku_indices % 8

    g.ndata['row'] = torch.tensor(rows, dtype=torch.long).to(device)
    g.ndata['col'] = torch.tensor(cols, dtype=torch.long).to(device)
    g.ndata['q'] = torch.tensor(puzzle, dtype=torch.long).to(device)
    g.ndata['a'] = torch.tensor(puzzle, dtype=torch.long).to(device)

    pred, _ = model(g, False)
    pred = pred.cpu().data.numpy().reshape([64])
    return pred