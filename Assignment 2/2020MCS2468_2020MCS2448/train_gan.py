import argparse


# python3 train_gan.py --train_dir <directory_containing_query_and_target_sudoku_boards> --output_file part1.pth --sample_file <path_to_sample_images.npy>

my_parser = argparse.ArgumentParser(allow_abbrev=False)

my_parser.add_argument('--train_dir', required=True, type=str, action='store')

my_parser.add_argument('--output_file', required=True, type=str, action='store')

my_parser.add_argument('--sample_file', required=True, type=str, action='store')

args = my_parser.parse_args()
