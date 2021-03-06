from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Simple QA BiGRU model - Ferhan')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_embedding', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=400)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test', action='store_true', dest='test', help='turn on test mode; no training.')
    parser.add_argument('--not_bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--clip', type=float, default=0.4, help='gradient clipping')
    parser.add_argument('--log_every', type=int, default=400)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducing results')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--device', type=int, default=0, help='GPU device to use')
    parser.add_argument('--save_path', type=str, default='saved_checkpoints')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args
