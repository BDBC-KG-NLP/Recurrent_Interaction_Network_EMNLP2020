import argparse

# training configurations
parser = argparse.ArgumentParser(description="RIN model for nyt and webnlg")

parser.add_argument('--dataset', type=str, default='nyt', help='dataset: nyt or webnlg')
parser.add_argument("--batch_size", type=int, default=50, help="batch size")
parser.add_argument("--optimizer", type=str, default="adam", help="optimizer (or, trainer)")
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--n_epoch", type=int, default=100, help="number of training epoch")
parser.add_argument("--early_stop", type=int, default=30, help="number of early stop epochs")
parser.add_argument("--dropout_rate", type=float, default=0.1, help="dropout rate for the embedding layer")
parser.add_argument('--rounds', type=int, default=4, help='Number of rounds.')
parser.add_argument('--exact_match', default=False, help='If use exact match evaluation.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--print_step', type=int, default=100, help='Print log every k steps in training process.')

# network configurations
parser.add_argument("--dim_bilstm_hidden", type=int, default=50, help="hidden dimension for the bilstm")
parser.add_argument("--dim_w", type=int, default=100, help="word embedding dimension")
parser.add_argument("--dim_pos", type=int, default=10, help="pos embedding dimension")

args = parser.parse_args()
