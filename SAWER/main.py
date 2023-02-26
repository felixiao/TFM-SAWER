import torch
import yaml
import argparse

from models import Trainer
from utils.data_utils import now_time, init_seed
from utils.constants import *


def main():
    parser = argparse.ArgumentParser(description='PErsonalized Transformer for Explainable Recommendation (PETER)')
    parser.add_argument('--model_name', type=str, default='peter',
                        help='model name (peter, sequer, etc)')
    parser.add_argument('--model_suffix', type=str, default='',
                        help='model suffix for different model configs')
    parser.add_argument('--dataset', type=str, default=None,
                        help='model name (amazon_movies, yelp, tripadvisor)')
    parser.add_argument('--fold', type=str, default=0,
                        help='data partition index')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--test', action='store_true',
                        help='use a small fraction of the data')
    parser.add_argument('--no-generate', action='store_true',
                        help='use a small fraction of the data')
    parser.add_argument('--check-convergence', action='store_true',
                        help='use same batch always to check convergence of model')
    # parser.add_argument('--log_interval', type=int, default=200,
    #                     help='report interval')
    parser.add_argument('--seed', type=int, default=RNG_SEED,
                        help='seed for reproducibility')
    args = parser.parse_args()
    if args.dataset is None:
        parser.error('--dataset should be provided for loading data')
    elif args.dataset not in DATA_PATHS:
        parser.error(f'--dataset supported values are: {", ".join(list(DATA_PATHS.keys()))} -- Provided value: {args.dataset}')

    with open(os.path.join(CONFIG_PATH, f'{args.model_name}{args.model_suffix}.yaml'), 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    # Set the random seed manually for reproducibility.
    init_seed(args.seed, reproducibility=True)
    if torch.cuda.is_available():
        if not args.cuda:
            print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
    device = torch.device('cuda' if args.cuda else 'cpu')

    if not os.path.exists(CKPT_PATH):
        os.makedirs(CKPT_PATH)
    pred_file = PRED_F.split('.')
    pred_file[0] += f'_{args.dataset}_{args.fold}_{args.model_name}{args.model_suffix}'
    prediction_path = os.path.join(CKPT_PATH, '.'.join(pred_file))

    if args.test:
        cfg['epochs'] = 1

    if args.check_convergence:
        args.test = True
        args.log_interval = 1
    args.no_generate = args.no_generate | args.test

    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        print('{:40} {}'.format(arg, getattr(args, arg)))
    for arg in cfg:
        print('{:40} {}'.format(arg, cfg[arg]))
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)

    trainer = Trainer(args.dataset, args.fold, prediction_path, args.model_name, args.model_suffix, cfg, device,
                      args.test, not args.no_generate, args.check_convergence)
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    main()

    # proc_data = Preprocess_Data(data_file='FMLP/data/reviews_new.pickle',index_dir='FMLP/data')
    # proc_data.load_file()
    # print(proc_data.peter_data.columns)
    # proc_data.convert_index()
    # proc_data.select_data_num_history(limit=5)
    # proc_data.split_data()