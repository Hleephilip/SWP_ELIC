import argparse
import os
import sys
from pathlib import Path
from utils.seed_init import seed_fix
from utils.eval_ensemble_model import evaluate

def init_parse():
    parser = argparse.ArgumentParser(description='Evaluation a trained model for leaf classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data-path', type=Path, default='./data/')
    parser.add_argument('--save-dir-1', type=Path, default='DenseNet169_1st')
    parser.add_argument('--save-dir-2', type=Path, default='SEDenseNet169_1st')
    parser.add_argument('--save-dir-3', type=Path, default='ResNeXt-101-32x8d-Finetune_1st')
    parser.add_argument('--num_label', type=int, default=179)
    parser.add_argument('--use-pretrained', type=int, default=1)

    return parser.parse_args(args=[])

if __name__ == '__main__' :
    seed_fix(10)
    args = init_parse()
    args.check_dir_1 = './result' / args.save_dir_1
    args.check_dir_2 = './result' / args.save_dir_2
    args.check_dir_3 = './result' / args.save_dir_3
    evaluate(args)