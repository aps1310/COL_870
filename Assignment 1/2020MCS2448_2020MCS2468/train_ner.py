import argparse
from cifar.dataset import get_default_device
from ner.dataset import get_data_loaders
from ner.model import BiLSTM, BiLSTM_CRF
from ner.train import train, train_crf

# python3 train_ner.py --initialization glove --char_embeddings 1  --layer_normalization 1  --crf  1
# --output_file trained_models_ner/part_2.2_crf_glove_char_ln.pth --data_dir ../data/ner-gmb
# --glove_embeddings_file ../data/glove/glove.6B.100d.txt
# --vocabulary_output_file trained_models_ner/part_2.2_crf_glove_char_ln.vocab

my_parser = argparse.ArgumentParser(allow_abbrev=False)

my_parser.add_argument('--initialization', required=True, type=str, action='store')

my_parser.add_argument('--char_embeddings', required=True, type=int, action='store', choices=(0, 1))

my_parser.add_argument('--layer_normalization', required=True, type=int, action='store', choices=(0, 1))

my_parser.add_argument('--crf', required=True, type=int, action='store', choices=(0, 1))

my_parser.add_argument('--output_file', required=True, type=str, action='store')

my_parser.add_argument('--data_dir', required=True, type=str, action='store')

my_parser.add_argument('--glove_embeddings_file', required=False, type=str, action='store')

my_parser.add_argument('--vocabulary_output_file', required=True, type=str, action='store')

args = my_parser.parse_args()
args.char_embeddings = True if args.char_embeddings == 1 else False
args.layer_normalization = True if args.layer_normalization == 1 else False

device = get_default_device()

train_loader, val_loader, test_loader, data_stat = get_data_loaders(args.data_dir, device,
                                                                    vocab_out_file=args.vocabulary_output_file,
                                                                    is_cleaned=False)
vocab_size = data_stat['vocab_size']
char_vocab_size = data_stat['char_vocab_size']
num_tags = data_stat['tag_size']
tag2idx = data_stat['tag2idx']
idx2tag = data_stat['idx2tag']

if args.crf != 1:
    if args.initialization == 'glove':
        bilstm_model = BiLSTM(vocab_size, char_vocab_size, num_tags, enable_char=args.char_embeddings,
                              embFileName=args.glove_embeddings_file, enable_layer_norm=args.layer_normalization)
    else:
        bilstm_model = BiLSTM(vocab_size, char_vocab_size, num_tags, enable_char=args.char_embeddings,
                              enable_layer_norm=args.layer_normalization)
    model = bilstm_model.to(device)
    train(model, train_loader, val_loader, model_save_path=args.output_file, idx2tag=idx2tag, already_trained=False)

else:
    if args.initialization == 'glove':
        bilstm_crf_model = BiLSTM_CRF(vocab_size, char_vocab_size, num_tags, enable_char=args.char_embeddings,
                                      embFileName=args.glove_embeddings_file, enable_layer_norm=args.layer_normalization)
    else:
        bilstm_crf_model = BiLSTM_CRF(vocab_size, char_vocab_size, num_tags, enable_char=args.char_embeddings,
                                      enable_layer_norm=args.layer_normalization)
    model = bilstm_crf_model.to(device)
    train_crf(model, train_loader, val_loader, model_save_path=args.output_file, idx2tag=idx2tag, already_trained=False)
