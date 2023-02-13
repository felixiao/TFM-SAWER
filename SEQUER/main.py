import torch
import yaml
import argparse
import logging

from trainer import Trainer
from utils import now_time, init_seed, load_data,log_info
from constants import *
import torch.multiprocessing as mp

from torch.distributed import init_process_group, destroy_process_group

def main():
    parser = argparse.ArgumentParser(description='PErsonalized Transformer for Explainable Recommendation (PETER)')
    parser.add_argument('--model_name', type=str, default='peter',
                        help='model name (peter, sequer, etc)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='model name (amazon-movies, yelp, tripadvisor)')
    parser.add_argument('--fold', type=str, default=0,
                        help='data partition index')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--test', action='store_true',
                        help='use a small fraction of the data')
    parser.add_argument('--no-generate', action='store_true',
                        help='use a small fraction of the data')
    parser.add_argument('--seed', type=int, default=RNG_SEED,
                        help='seed for reproducibility')
    parser.add_argument('--no_train',action='store_true',
                        help='Skip Training, test the saved model')
    parser.add_argument('--log_file',type=str, default='./Log/SEQ-FMLP_PETER.log',
                        help='the log file')

    args = parser.parse_args()
    if args.dataset is None:
        parser.error('--dataset should be provided for loading data')
    elif args.dataset not in DATA_PATHS:
        parser.error(f'--dataset supported values are: {", ".join(list(DATA_PATHS.keys()))}')

    with open(os.path.join(CONFIG_PATH, f'{args.model_name}.yaml'), 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    log_info(f'{"ARGUMENTS":-^80}',gpu_id = int(os.environ["LOCAL_RANK"]),time=False,gpu=False)
    for arg in vars(args):
        log_info(f'{arg:<40} {getattr(args, arg)}',gpu_id = int(os.environ["LOCAL_RANK"]),time=False,gpu=False)
        # print('{:40} {}'.format(arg, getattr(args, arg)))
    log_info(f'{"CONFIGS":-^80}',gpu_id = int(os.environ["LOCAL_RANK"]),time=False,gpu=False)
    for arg in cfg:
        log_info(f'{arg:<40} {cfg[arg]}',gpu_id = int(os.environ["LOCAL_RANK"]),time=False,gpu=False)
        # print('{:40} {}'.format(arg, cfg[arg]))
    log_info(f'{"ARGUMENTS":-^80}',gpu_id = int(os.environ["LOCAL_RANK"]),time=False,gpu=False)

    # Set the random seed manually for reproducibility.
    init_seed(args.seed, reproducibility=True)
    if torch.cuda.is_available():
        if not args.cuda:
            logging.info('WARNING: You have a CUDA device, so you should probably run with --cuda',time=False)
            # print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
    device = torch.device('cuda' if args.cuda else 'cpu')

    if not os.path.exists(CKPT_PATH):
        os.makedirs(CKPT_PATH)
    pred_file = PRED_F.split('.')
    pred_file[0] += f'_{args.dataset}_{args.fold}_{args.model_name}'
    prediction_path = os.path.join(CKPT_PATH, '.'.join(pred_file))

    args.no_generate = args.no_generate | args.test

    world_size = torch.cuda.device_count()
    # print(f'World_Size: {world_size}')
    log_info(f'World_Size(Num GPUs): {world_size}',gpu_id = int(os.environ["LOCAL_RANK"]))
    
    # mp.spawn(train_model, args=(args,prediction_path,cfg,device), nprocs=world_size)
    train_model(args,prediction_path,cfg,device)
    # try: 
    #     destroy_process_group()  
    # except : 
    #     os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
    
def train_model(args,prediction_path,cfg,device):
    ddp_setup()
    index_dir = os.path.join(DATA_PATHS[args.dataset], str(args.fold))
    data_path = os.path.join(DATA_PATHS[args.dataset], 'reviews_new.pickle')
    seq_mode = cfg.get('seq_mode', 0)
    batch_size = cfg.get('batch_size', 128)
    vocab_size = cfg.get('vocab_size', 5000)
    hist_len = int(cfg.get('hist_len',0))
    word_len = int(cfg.get('text_len',15))
    pre_trained = cfg.get('pre_train', False)
    load_direct = cfg.get('load_direct',True)

    data_helper = load_data(DATA_PATHS[args.dataset], index_dir, seq_mode,batch_size, vocab_size,hist_len,word_len,test_flg=args.test,load_direct=load_direct,pre_trained=pre_trained)
    trainer = Trainer(args.dataset, args.fold, prediction_path, args.model_name, data_helper,cfg, device,
                      args.test, not args.no_generate,0,torch.cuda.device_count())
    
    if not args.no_train:
        trainer.train()
    trainer.test()
    destroy_process_group()
    
def ddp_setup():
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    # if 'NUMEXPR_MAX_THREADS' in os.environ: 
    #     print(f'[GPU {rank}] NUMEXPR_MAX_THREADS: { os.environ["NUMEXPR_MAX_THREADS"]}') 
    #     os.environ.pop('NUMEXPR_MAX_THREADS')
    # if "OMP_NUM_THREADS" in os.environ:
    #     print(f'[GPU {rank}] OMP_NUM_THREADS: { os.environ["OMP_NUM_THREADS"]}')
    #     os.environ.pop('OMP_NUM_THREADS')
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = 'INFO' # 'DETAIL'
    os.environ["TORCH_CPP_LOG_LEVEL"]="WARNING" # 'INFO'

    init_process_group(backend="nccl")

if __name__ == '__main__':
    main()

    # proc_data = Preprocess_Data(data_file='FMLP/data/reviews_new.pickle',index_dir='FMLP/data')
    # proc_data.load_file()
    # print(proc_data.peter_data.columns)
    # proc_data.convert_index()
    # proc_data.select_data_num_history(limit=5)
    # proc_data.split_data()