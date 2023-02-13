import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import trange,tqdm
import random

from models import FMLPRecModel
from trainers import FMLPRecTrainer
from utils import EarlyStopping, check_path, set_seed, get_local_time, get_seq_dic, get_dataloder, get_rating_matrix,get_user_seqs_and_sample
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# load data
def LoadData():
    data_file = './FMLP/data/Movie_and_TV_GT5.pickle'
    peter_data = pd.DataFrame.from_records(pd.read_pickle(data_file))
    user_seq = peter_data['hist_item_index']
    sample_seq = peter_data['neg_sample_index']

    num_users = len(peter_data)
    max_item = len(peter_data['item'].unique())

    train_dataset = FMLPRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = FMLPRecDataset(args, user_seq, test_neg_items=sample_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = FMLPRecDataset(args, user_seq, test_neg_items=sample_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size)
    word2idx = corpus.word_dict.word2idx
    idx2word = corpus.word_dict.idx2word
    feature_set = corpus.feature_set
    train_data = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True)
    val_data = Batchify(corpus.valid, word2idx, args.words, args.batch_size)
    test_data = Batchify(corpus.test, word2idx, args.words, args.batch_size)

    return train_dataloader, eval_dataloader, test_dataloader

def BuildModel():
    model = FMLPRecModel(args=args)

def TrainModel():
    trainer = FMLPRecTrainer(model, train_dataloader, eval_dataloader,test_dataloader, args)
    early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
    for epoch in range(args.epochs):
        trainer.train(epoch)
        scores, _ = trainer.valid(epoch, full_sort=args.full_sort)
        # evaluate on MRR
        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("---------------Sample 99 results---------------")
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0, full_sort=args.full_sort)