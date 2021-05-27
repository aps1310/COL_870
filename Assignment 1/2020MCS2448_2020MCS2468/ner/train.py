# Imports
import torch
import torch.optim as optim
from seqeval.metrics import accuracy_score, f1_score
from seqeval.scheme import IOB2
import time


def get_stats(model, data_loader, idx2tag):
    y_pred = []
    y_true = []

    for i, xy in enumerate(data_loader):
        sentences = xy[0]
        tags = xy[1]
        chars = xy[2]

        chars_len = []
        batch_size = len(sentences)

        if model.enable_char:
            tag_out = model(sentences, chars, chars_len)
        else:
            tag_out = model(sentences)

        tag_out = tag_out.view(batch_size, -1, tag_out.shape[1])
        tag_out = torch.max(tag_out, dim=2).indices

        for i, tag in enumerate(tags):
            x1 = []
            x2 = []
            for j, y in enumerate(tag):
                if y.item() == 17:
                    break
                else:
                    x1.append(idx2tag[y.item()])
                    x2.append(idx2tag[tag_out[i][j].item()])
            y_true.append(x1)
            y_pred.append(x2)

    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, scheme=IOB2), \
           f1_score(y_true, y_pred, average='macro', scheme=IOB2)


# def loss_fn(outputs, labels, batch_size, seq_len):
#     loss_function = nn.CrossEntropyLoss(ignore_index=17, reduction='sum')
#     outputs = outputs.contiguous().view(batch_size * seq_len, -1)
#     total_loss = loss_function(outputs, labels.contiguous().view(batch_size * seq_len))
#     return total_loss

def loss_fn(outputs, labels):
    labels = labels.view(-1)
    mask = (labels != 17).float()

    labels[mask == False] = 0

    num_tokens = torch.sum(mask).item()

    outputs = outputs[range(outputs.shape[0]), labels]*mask

    return -torch.sum(outputs)/num_tokens


def train(model, train_loader, val_loader, model_save_path, idx2tag, already_trained=False,
          learning_rate=0.01, weight_decay=0.0001):
    train_accuracy_history, train_micro_f1_history, train_macro_f1_history = [], [], []
    val_accuracy_history, val_micro_f1_history, val_macro_f1_history = [], [], []

    if already_trained:
        model.load_state_dict(torch.load(model_save_path))
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        epoch_start_time = time.time()
        for epoch in range(100):
            model.train()
            # Used to calculate stats for train set
            y_pred = []
            y_true = []
            for i, xy in enumerate(train_loader):
                sentences = xy[0]
                tags = xy[1]
                chars = xy[2]

                chars_len = []
                batch_size = sentences.size(0)

                model.zero_grad()

                if model.enable_char:
                    tag_scores = model(sentences, chars, chars_len)
                else:
                    tag_scores = model(sentences)


                loss = loss_fn(tag_scores, tags)
                # get_stats(model, val_loader, idx2tag)
                tag_scores = tag_scores.view(batch_size, -1, tag_scores.shape[1])
                tag_out = torch.max(tag_scores, dim=2).indices
                # tags = tags.view(tags.shape[0]*tags.shape[1])
                for i, tag in enumerate(tags):
                    x1 = []
                    x2 = []
                    for j, y in enumerate(tag):
                        if y.item() == 17:
                            break
                        else:
                            x1.append(idx2tag[y.item()])
                            x2.append(idx2tag[tag_out[i][j].item()])
                    y_true.append(x1)
                    y_pred.append(x2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.gradientClipingValue)
                optimizer.step()


            print("{}/{} epochs completed in ".format(epoch + 1, 100, time.time()-epoch_start_time))
            print("Calculating training stats")
            train_accuracy, train_micro_f1, train_macro_f1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred,
                                                                                                      scheme=IOB2), f1_score(
                y_true, y_pred, average='macro', scheme=IOB2)
            print("Calculating validation stats")
            val_accuracy, val_micro_f1, val_macro_f1 = get_stats(model, val_loader, idx2tag)
            train_accuracy_history.append(train_accuracy)
            train_micro_f1_history.append(train_micro_f1)
            train_macro_f1_history.append(train_macro_f1)

            val_accuracy_history.append(val_accuracy)
            val_micro_f1_history.append(val_micro_f1)
            val_macro_f1_history.append(val_macro_f1)

    return train_accuracy_history, train_micro_f1_history, train_macro_f1_history, val_accuracy_history, val_micro_f1_history, val_macro_f1_history


def train_crf(model, train_loader, val_loader, model_save_path, idx2tag, already_trained=False,
              learning_rate=0.01, weight_decay=0.0001):
    train_accuracy_history, train_micro_f1_history, train_macro_f1_history = [], [], []
    val_accuracy_history, val_micro_f1_history, val_macro_f1_history = [], [], []

    if already_trained:
        model.load_state_dict(torch.load(model_save_path))
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(100):
            for i, xy in enumerate(train_loader):
                sentences = xy[0]
                tags = xy[1]
                mask = tags != len(idx2tag)
                chars = xy[2]

                chars_len = []
                for char in chars:
                    x = []
                    for i in range(char.shape[0]):
                        x.append(torch.count_nonzero(char[i]).item())
                    chars_len.append(x)

                model.zero_grad()

                loss = model.neg_log_likelihood(sentences, tags, mask, chars=chars, chars_length=chars_len)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.gradientClipingValue)
                optimizer.step()

            print("{}/{} epochs completed".format(epoch + 1, 100))
            print("Calculating training stats")
            train_accuracy, train_micro_f1, train_macro_f1 = get_stats(model, train_loader, idx2tag)
            print("Calculating validation stats")
            val_accuracy, val_micro_f1, val_macro_f1 = get_stats(model, val_loader, idx2tag)
            train_accuracy_history.append(train_accuracy)
            train_micro_f1_history.append(train_micro_f1)
            train_macro_f1_history.append(train_macro_f1)

            val_accuracy_history.append(val_accuracy)
            val_micro_f1_history.append(val_micro_f1)
            val_macro_f1_history.append(val_macro_f1)

        return train_accuracy_history, train_micro_f1_history, train_macro_f1_history,\
            val_accuracy_history, val_micro_f1_history, val_macro_f1_history
