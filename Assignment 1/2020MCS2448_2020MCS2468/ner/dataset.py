import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# Converts Sentence into vector.
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[word] for word in seq]
    return torch.tensor(idxs, dtype=torch.long)


def prepare_sequence_chars(seq, to_ix):
    idxs = [torch.tensor([to_ix[char] for char in word]) for word in seq]
    return pad_sequence(idxs, batch_first=False).T


def get_dict_map(data, token_or_tag):
    if token_or_tag == 'token':
        vocab = list(set(data['TOKEN'].to_list()))
    elif token_or_tag == 'char':
        vocabSet = set()
        for word in (data['TOKEN'].to_list()):
            vocabSet = vocabSet.union(list(word))
        vocab = list(vocabSet)
    else:
        vocab = list(set(data['TAG'].to_list()))

    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok


class GMBDataset(Dataset):
    def __init__(self, txt_file_path, is_cleaned=False):

        def add_sentence_no(df):
            df['Sentence#'] = 0
            df.at[0, 'Sentence#'] = 1
            cur_sentence = 1
            for index, row in df.iterrows():
                df.at[index, 'Sentence#'] = cur_sentence
                if row['TOKEN'] == '.':
                    cur_sentence += 1

        if is_cleaned:
            dataframe = pd.read_csv(txt_file_path, sep=',')
        else:
            dataframe = pd.read_csv(txt_file_path, sep=' ', names=['WORD', 'POS', 'TOKEN', 'TAG']).drop(columns=['WORD', 'POS'])
            add_sentence_no(dataframe)

        dataframe.dropna(inplace=True)
        self.token2idx = None
        self.tag2idx = None
        self.char2idx = None
        self.df = dataframe

    def __len__(self):
        return self.df['Sentence#'].max()

    def __getitem__(self, idx):
        idx_sentence = self.df.loc[self.df['Sentence#'] == idx + 1]
        tokens = []
        tags = []
        chars = []
        for index, row in idx_sentence.iterrows():
            tokens.append(row['TOKEN'])
            tags.append(row['TAG'])
            chars.append(list(row['TOKEN']))
        tokens = prepare_sequence(tokens, self.token2idx)
        tags = prepare_sequence(tags, self.tag2idx)
        chars = prepare_sequence_chars(chars, self.char2idx)

        return tokens, tags, chars


def get_data_loaders(data_path, device, vocab_out_file, is_cleaned=False):
    print("Loading and cleaning data. This might take a minute or two.....")
    if is_cleaned:
        trainset = GMBDataset(data_path + '/cleaned_train.csv', is_cleaned=True)
        valset = GMBDataset(data_path + '/cleaned_val.csv', is_cleaned=True)
        testset = GMBDataset(data_path + '/cleaned_test.csv', is_cleaned=True)
    else:
        trainset = GMBDataset(data_path + '/train.txt')
        valset = GMBDataset(data_path + '/dev.txt')
        testset = GMBDataset(data_path + '/test.txt')

    token2idx, idx2token = get_dict_map(pd.concat([trainset.df, testset.df, valset.df]), 'token')
    char2idx, idx2char = get_dict_map(pd.concat([trainset.df, testset.df, valset.df]), 'char')
    tag2idx, idx2tag = get_dict_map(pd.concat([trainset.df, testset.df, valset.df]), 'tag')

    num_tags = len(tag2idx)
    num_char = len(char2idx)

    trainset.token2idx, trainset.tag2idx, trainset.char2idx = token2idx, tag2idx, char2idx
    valset.token2idx, valset.tag2idx, valset.char2idx = token2idx, tag2idx, char2idx
    testset.token2idx, testset.tag2idx, testset.char2idx = token2idx, tag2idx, char2idx

    def collate(batch):
        x = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=num_tags)
        y = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=num_tags)
        z = []
        max_word_len_in_batch = -1
        for b in batch:
            max_word_len_in_batch = max(b[2].shape[1], max_word_len_in_batch)
        for b in batch:
            char = b[2]
            m = nn.ConstantPad2d((0, max_word_len_in_batch - char.shape[1], 0, y.shape[1] - char.shape[0]), value=num_char)
            char = m(char)
            z.append(char)

        z = torch.stack(z)
        return x, y, z

    train_loader = DataLoader(trainset, batch_size=128,
                              shuffle=True, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(testset, batch_size=128,
                             shuffle=False, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(valset, batch_size=128,
                            shuffle=True, num_workers=0, collate_fn=collate)

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    data_stat = {'vocab_size': len(token2idx), 'char_vocab_size': len(char2idx), 'tag_size': len(tag2idx),
                 'idx2tag': idx2tag, 'tag2idx': tag2idx,
                 'char2idx': char2idx, 'idx2char': idx2char,
                 'token2idx': token2idx, 'idx2token': idx2token}
    with open(vocab_out_file, 'wb') as f:
        pickle.dump(data_stat, f)
    return train_loader, val_loader, test_loader, data_stat
