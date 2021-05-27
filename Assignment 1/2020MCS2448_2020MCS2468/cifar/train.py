# Imports
import torch
import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from sklearn import metrics
from dataset import to_device


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='cifar/model/checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train(device,model, train_loader, val_loader, model_save_path='cifar/model/checkpoint.pt', already_trained=False,
          learning_rate=0.1, momentumValue=0.9, wieghtDecayValue=0.0001):
    # Used to handle special case of Batch-Instance Normalization
    # As mentioned in the paper, we need to clip the gate value to be in range(0, 1)
    bin_gates = [p for p in model.parameters() if getattr(p, 'bin_gate', False)]
    if already_trained:
        model.load_state_dict(torch.load(model_save_path))
    else:
        criterion = CrossEntropyLoss()

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentumValue,
                              weight_decay=wieghtDecayValue)

        early_stopping = EarlyStopping(patience=20, verbose=True, path=model_save_path)

        train_loss_history = []
        train_acc_history = []
        train_f1_micro_history = []
        train_f1_macro_history = []

        val_loss_history = []
        val_acc_history = []
        val_f1_micro_history = []
        val_f1_macro_history = []

        for epoch in range(100):  # loop over the dataset multiple times

            running_loss = 0.0
            actual_labels = []
            predicted_labels = []

            model.train()
            for i, data in enumerate(train_loader, 1):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                actual_labels = np.hstack((actual_labels, labels.cpu().numpy()))
                predicted_labels = np.hstack((predicted_labels, predicted.cpu().numpy()))

                loss.backward()
                optimizer.step()
                for p in bin_gates:
                    p.data.clamp_(min=0, max=1)

                running_loss += loss.item()

            train_loss = running_loss / i
            train_acc = metrics.accuracy_score(actual_labels, predicted_labels, normalize=True)
            train_f1_micro = metrics.f1_score(actual_labels, predicted_labels, average='micro')
            train_f1_macro = metrics.f1_score(actual_labels, predicted_labels, average='macro')

            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            train_f1_micro_history.append(train_f1_micro)
            train_f1_macro_history.append(train_f1_macro)

            print('[%d] Train loss: %.5f   Train Accuracy: %.5f  Train f1_micro: %.5f  Train f1_macro: %.5f ' % (
            epoch + 1, train_loss, train_acc, train_f1_micro, train_f1_macro))

            # Run Validation
            model.eval()  # prep model for evaluation
            with torch.no_grad():  # prevent affecting gradients
                running_loss = 0.0
                actual_labels = []
                predicted_labels = []
                quantile_1 = []
                quantile_2 = []
                quantile_80 = []
                quantile_99 = []

                features = None
                for i, data in enumerate(val_loader, 1):
                    inputs, labels = data
                    outputs = model(inputs)
                    if features is None:
                        features = model.penultimate_layer_activation
                    else:
                        features = torch.cat((features, model.penultimate_layer_activation),dim=0)

                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    actual_labels = np.hstack((actual_labels, labels.cpu().numpy()))
                    predicted_labels = np.hstack((predicted_labels, predicted.cpu().numpy()))

                    # record validation loss

                    running_loss += loss.item()

                features = torch.flatten(features)
                q = torch.tensor((0.01, 0.2, 0.8, 0.99))
                q = to_device(q, device)
                qt = torch.quantile(features, q)

            quantile_1.append(qt[0])
            quantile_2.append(qt[1])
            quantile_80.append(qt[2])
            quantile_99.append(qt[3])

            val_loss = running_loss / i
            val_acc = metrics.accuracy_score(actual_labels, predicted_labels, normalize=True)
            val_f1_micro = metrics.f1_score(actual_labels, predicted_labels, average='micro')
            val_f1_macro = metrics.f1_score(actual_labels, predicted_labels, average='macro')

            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            val_f1_micro_history.append(val_f1_micro)
            val_f1_macro_history.append(val_f1_macro)
            print(
                '[%d] Validation loss: %.5f   Validation Accuracy: %.5f  Validation f1_micro: %.5f  Validation f1_macro: %.5f ' % (
                epoch + 1, val_loss, val_acc, val_f1_micro, val_f1_macro))

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            print()

        print('Finished Training')
        return (train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_f1_micro_history,
                val_f1_micro_history, train_f1_macro_history, val_f1_macro_history, quantile_1, quantile_2, quantile_80,
                quantile_99)




