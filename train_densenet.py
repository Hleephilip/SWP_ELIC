import argparse
import os
import sys
from pathlib import Path
from utils.seed_init import seed_fix
from utils.train_densenet_model import train

def init_parse():
    parser = argparse.ArgumentParser(description='Training a model for leaf classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use-cutmix', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--data-path', type=Path, default='./data/')
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--log-print-interval', type=int, default=1)
    parser.add_argument('--save-dir', type=Path, default='DenseNet169')
    parser.add_argument('--use-pretrained', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--cutmix_prob', type=float, default=0.5)
    parser.add_argument('--num_label', type=int, default=179)

    return parser.parse_args(args=[])

if __name__ == '__main__' :
    seed_fix(10)

    args = init_parse()
    args.check_dir = './result' / args.save_dir
    args.check_dir.mkdir(parents=True, exist_ok=True)

    train(args)