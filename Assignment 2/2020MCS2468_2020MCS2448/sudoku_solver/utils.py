import numpy as np
import torch


def create_constraint_mask():
    constraint_mask = torch.zeros((64, 3, 64), dtype=torch.float)
    # row constraints
    for a in range(64):
        r = 8* (a // 8)
        for b in range(8):
            constraint_mask[a, 0, r + b] = 1
    # column constraints
    for a in range(64):
        c = a % 8
        for b in range(8):
            constraint_mask[a, 1, c + 8 * b] = 1
    # box constraints
    for a in range(64):
        r = a // 8
        c = a % 8
        br = r//2
        bc = c//4
        for b in range(8):
            r = b // 4
            constraint_mask[a, 2, 16*br+4*bc+b+r*4] = 1
    return constraint_mask


mask = create_constraint_mask().numpy().astype(np.uint8)


def is_valid(sudoku):
    for i in range(64):
        num = sudoku[i]
        if num == 0:
            continue
        for j in range(3):
            neighbour = [a*b for a, b in zip(sudoku, mask[i][j])]
            if neighbour.count(num) != 1:
                return False, num
    return True, 0


