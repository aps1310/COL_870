# python3 test_ner.py  --initialization glove --char_embeddings 1  --layer_normalization 1  --crf  1
# --model_file trained_models_ner/part_2.2_crf_glove_char_ln.pth --test_data_file ../data/ner_test_data.txt
# --output_file trained_models_ner/predictions_part_2.2_crf_glove_char_ln.txt
# --glove_embeddings_file ../data/glove/glove.6B.100d.txt
# --vocabulary_input_file trained_models_ner/part_2.2_crf_glove_char_ln.vocab
import argparse
import pickle

from cifar.dataset import get_default_device
from ner.model import BiLSTM, BiLSTM_CRF
from ner.test import test


my_parser = argparse.ArgumentParser(allow_abbrev=False)

my_parser.add_argument('--initialization', required=True, type=str, action='store')

my_parser.add_argument('--char_embeddings', required=True, type=int, action='store', choices=(0, 1))

my_parser.add_argument('--layer_normalization', required=True, type=int, action='store', choices=(0, 1))

my_parser.add_argument('--crf', required=True, type=int, action='store', choices=(0, 1))

my_parser.add_argument('--model_file', required=True, type=str, action='store')

my_parser.add_argument('--test_data_file', required=True, type=str, action='store')

my_parser.add_argument('--output_file', required=True, type=str, action='store')

my_parser.add_argument('--glove_embeddings_file', required=False, type=str, action='store')

my_parser.add_argument('--vocabulary_input_file', required=True, type=str, action='store')

args = my_parser.parse_args()
args.char_embeddings = True if args.char_embeddings == 1 else False
args.layer_normalization = True if args.layer_normalization == 1 else False

device = get_default_device()

f = open(args.vocabulary_input_file, 'rb')
vocab = pickle.load(f)
vocab_size = vocab['vocab_size']
num_tags = vocab['tag_size']
num_chars = vocab['char_vocab_size']


if args.crf != 1:
    if args.initialization == 'glove':
        bilstm_model = BiLSTM(vocab_size, num_chars, num_tags, enable_char=args.char_embeddings,
                              embFileName=args.glove_embeddings_file, enable_layer_norm=args.layer_normalization)
    else:
        bilstm_model = BiLSTM(vocab_size, num_chars, num_tags, enable_char=args.char_embeddings,
                              enable_layer_norm=args.layer_normalization)
    model = bilstm_model.to(device)
    test(model, device, args.test_data_file, args.output_file, vocab)
else:
    if args.initialization == 'glove':
        bilstm_crf_model = BiLSTM_CRF(vocab_size, num_chars, num_tags, enable_char=args.char_embeddings,
                                      embFileName=args.glove_embeddings_file, enable_layer_norm=args.layer_normalization)
    else:
        bilstm_crf_model = BiLSTM_CRF(vocab_size, num_chars, num_tags, enable_char=args.char_embeddings,
                                      enable_layer_norm=args.layer_normalization)
    model = bilstm_crf_model.to(device)
    test(model, device, args.test_data_file, args.output_file, vocab)
