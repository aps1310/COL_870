import cv2
import numpy as np
import torch
from torchvision.transforms import transforms


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[27.67], std=[72.55]),
])

M, N = 28, 28


def classify(model, target_image, is_target=False):
    img = 255 - target_image
    digits = [img[x:x + M, y:y + N] for x in range(0, img.shape[0], M) for y in range(0, img.shape[1], N)]

    digits = np.array(digits)
    new_digits = torch.zeros(digits.shape[0], 1, digits.shape[1], digits.shape[2])
    for idx in range(len(digits)):
        new_digits[idx, :, :, :] = test_transform(digits[idx]).reshape(1, 1, 28, 28)

    pred = model(new_digits.cuda())
    if is_target:
        new_pred = pred[:, 1:10]
        sudoku = torch.argmax(new_pred, dim=1)+1
    else:
        sudoku = torch.argmax(pred, dim=1)
    sudoku = sudoku.tolist()
    return sudoku
