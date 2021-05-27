
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import pandas as pd
from cifar.dataset import DeviceDataLoader
from ner.dataset import GMBDataset


def get_test_dataloader(file_path, device, vocab):

    testset = GMBDataset(file_path)
    testset.token2idx = vocab['token2idx']
    testset.tag2idx = vocab['tag2idx']
    testset.char2idx = vocab['char2idx']
    num_tags = vocab['tag_size']
    num_chars = vocab['char_vocab_size']

    def collate(batch):
        x = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=vocab['vocab_size'])
        y = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=num_tags)
        z = []
        max_word_len_in_batch = -1
        for b in batch:
            max_word_len_in_batch = max(b[2].shape[1], max_word_len_in_batch)
        for b in batch:
            char = b[2]
            m = nn.ConstantPad2d((0, max_word_len_in_batch - char.shape[1], 0, y.shape[1] - char.shape[0]), value=num_chars)
            char = m(char)
            z.append(char)

        z = torch.stack(z)
        return x, y, z

    test_loader = DataLoader(testset, batch_size=128,
                             shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DeviceDataLoader(test_loader, device)

    return test_loader, vocab['idx2tag']


def test(model, device, filepath, output_file_path, input_vocab):

    test_loader, idx2tag = get_test_dataloader(filepath, device, input_vocab)

    dataframe = pd.read_csv(filepath, sep=' ', header=None)

    with torch.no_grad():
        model.eval()
        row = 0
        for index, data in enumerate(test_loader):
            print(index)
            sentences = data[0]
            mask = (sentences != 39421)
            chars = data[2]
            batch_size = sentences.size(0)

            if model.enable_char:
                tag_scores = model(sentences, chars, [])
            else:
                tag_scores = model(sentences)

            tag_scores = tag_scores.view(batch_size, -1, tag_scores.shape[1])
            tag_out = torch.max(tag_scores, dim=2).indices

            for i, tag in enumerate(tag_out):
                for j, y in enumerate(tag):
                    if not mask[i][j]:
                        break
                    else:
                        dataframe.at[row, 'O'] = idx2tag[tag_out[i][j].item()]
                        row += 1

    dataframe.to_csv(output_file_path)
