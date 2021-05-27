import torch
import numpy as np
from torch.optim import Adam


def train(model, train_loader, val_loader, device, epoch):
    opt = Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)

    for epoch in range(epoch):
        model.train()
        for i, g in enumerate(train_loader):
            g = g.to(device)
            _, loss = model(g)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 100 == 0:
                print(f"Epoch {epoch}, batch {i}, loss {loss.cpu().data}")

        print("Evaluating...")
        model.eval()
        dev_loss = []
        dev_res = []
        for g in val_loader:
            g = g.to(device)
            g.ndata['q'] = g.ndata['q'].to(device)
            g.ndata['a'] = g.ndata['a'].to(device)
            g.ndata['row'] = g.ndata['row'].to(device)
            g.ndata['col'] = g.ndata['col'].to(device)

            target = g.ndata['a']
            target = target.view([-1, 64])

            with torch.no_grad():
                preds, loss = model(g, is_training=False)
                preds = preds.view([-1, 64])

                for i in range(preds.size(0)):
                    dev_res.append(int(torch.equal(preds[i, :], target[i, :])))

                dev_loss.append(loss.cpu().detach().data)
        dev_acc = sum(dev_res) / len(dev_res)
        print(f"Dev loss {np.mean(dev_loss)}, accuracy {dev_acc}")

    return model
