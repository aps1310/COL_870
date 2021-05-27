from torch.optim import Adam
import os

from image_to_symbol.data import sudoku_dataloader2
from image_to_symbol.model import CNN_Sudoku
from utils import get_default_device


checkpoint_dir = ''
best_model_dir = ''

device = get_default_device()

model = CNN_Sudoku(num_steps=32, edge_drop=0.4)


if not os.path.exists('out'):
    os.mkdir('out')

print(model)
model.to(device)
trainloader, testloader, valloader = sudoku_dataloader2(75)

opt = Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)


epoch_num = 0
best_dev_acc = 0

for epoch in range(epoch_num, 100, 1):
    model.train()
    for i, (g, q_batch) in enumerate(trainloader):
        g = g.to(device)
        q_batch = q_batch.to(device)
        _, loss = model(g, q_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(f"Epoch {epoch}, batch {i}, loss {loss.cpu().data}")
