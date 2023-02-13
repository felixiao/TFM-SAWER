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


def listtostr(ls):
    lstr = ''
    for l in ls:
        lstr+=l+' '
    return lstr[:-1]

def get_negsample(row,uni_ui,uni_i,num_neg=99):
    list_neg = list(uni_i.difference(uni_ui[row['user']]))
    sample_list = random.sample(list_neg,num_neg)
    # print(len(set(sample_list)))
    return sample_list

def hist_item_to_index(row,item_dict):
    histitem_list = [item_dict[i] for i in row['hist_item']]
    histitem_list.append(row['item_index'])
    return histitem_list

def convert_pickle(limit=5):
    peter_data = pd.DataFrame.from_records(pd.read_pickle('./FMLP/data/reviews_new.pickle'))
    print(list(peter_data.columns))
    unique_users = list(peter_data['user'].unique())
    unique_items = set(peter_data['item'].unique())
    print('Unique/Total User:',len(unique_users),'/',len(peter_data['user']))
    print('Unique/Total Item:',len(unique_items),'/',len(peter_data['item']))
    user_itemset = {}
    
    item_dict = {i:id for id, i in enumerate(unique_items)}

    for u in tqdm(unique_users):
        df = peter_data[peter_data['user']==u]
        user_itemset[u] = set(df['item'].unique())
    print('neg_sample')
    peter_data['neg_sample'] = peter_data.apply(lambda row: get_negsample(row,user_itemset,unique_items), axis=1)
    print('item_index')
    peter_data['item_index'] = peter_data.apply(lambda row: item_dict[row['item']],axis=1)
    print('hist_item_index')
    peter_data['hist_item_index'] = peter_data.apply(lambda row:hist_item_to_index(row,item_dict),axis=1)
    print('neg_sample_index')
    peter_data['neg_sample_index'] = peter_data.apply(lambda row:[item_dict[i] for i in row['neg_sample']],axis=1)
    print('len_hist_item')
    peter_data['len_hist_item'] = peter_data.apply(lambda row:len(row['hist_item']),axis=1)
    peter_data = peter_data[peter_data['len_hist_item'] >= limit]
    peter_data.drop(['len_hist_item'],axis=1,inplace=True)

    peter_data.to_pickle(f'./FMLP/data/Movie_and_TV_GT{limit}.pickle')

    unique_users = list(peter_data['user'].unique())
    print('Unique/Total User:',len(unique_users),'/',len(peter_data['user']))

    for i in trange(3):
        p1=peter_data[peter_data['user']==unique_users[i]]
        p1 = p1[['user','unixReviewTime','item','item_index','hist_item','hist_item_index','neg_sample','neg_sample_index']]
        p1.sort_values(by=['user','unixReviewTime','item'],inplace=True)
        p1.to_csv(f'./FMLP/data/Movie_and_TV_example_{i}_user_{unique_users[i]}.csv',index=False)
    

def check_beauty():
    user_seq, _, _, _ = get_user_seqs_and_sample('data/Beauty.txt', 'data/Beauty_sample.txt')
    # seq_dic = {'user_seq':user_seq, 'num_users':num_users, 'sample_seq':sample_seq}
    _user_seq = []
    for id, seq in enumerate(user_seq[:10]):
        input_ids = seq[-52:-2]  # keeping same as train set
        print(id, input_ids)
        for i in range(1,len(input_ids)+1):
            print(i, input_ids[:i])
            _user_seq.append(input_ids[:i])


def remove_hist_lessthan(limit=5):
    peter_data = pd.DataFrame.from_records(pd.read_pickle('data/Movie_and_TV.pickle'))
    unique_users = list(peter_data['user'].unique())
    print('Unique/Total User:',len(unique_users),'/',len(peter_data['user']))

    peter_data['len_hist_item'] = peter_data.apply(lambda row:len(row['hist_item']),axis=1)

    # peter_data.drop(peter_data[peter_data['len_hist_item']<5].index,inplace=True)
    peter_data = peter_data[peter_data['len_hist_item'] >= limit]
    peter_data.drop(['len_hist_item'],axis=1,inplace=True)
    peter_data.to_pickle(f'data/Movie_and_TV_GT{limit}.pickle')
    unique_users = list(peter_data['user'].unique())
    print('Unique/Total User:',len(unique_users),'/',len(peter_data['user']))

    for i in trange(3):
        p1=peter_data[peter_data['user']==unique_users[i]]
        p1 = p1[['user','unixReviewTime','item','item_index','hist_item','hist_item_index','neg_sample','neg_sample_index']]
        p1.sort_values(by=['user','unixReviewTime','item'],inplace=True)
        p1.to_csv(f'data/Movie_and_TV_example_{i}_user_{unique_users[i]}.csv',index=False)

def check_csv():
    peter_data = pd.DataFrame.from_records(pd.read_pickle('data/Movie_and_TV.pickle'))
    # print(peter_data.dtypes)

    # peter_data = pd.read_csv('p1.csv',dtype={'user':object,'item':object,
    #     'rating':np.int64,'template':object,'predicted':object,
    #     'hist_item':object,'unixReviewTime':np.int64,'hist_review':object,'neg_sample':object})
    print(peter_data.dtypes)
    user_seq = []
    # print(peter_data['hist_item'][0])
    for id, seq in enumerate(peter_data['hist_item_index'][:10]):
        # print(seq)
        input_ids = seq[-52:-2]  # keeping same as train set
        print(id,input_ids)
        user_seq.append(input_ids)
        # for i in range(1,len(input_ids)+1):
        #     user_seq.append(input_ids[:i])
        #     print(i,input_ids[:i])
    print(user_seq)

# peter_data = pd.read_csv('./FMLP/data/Movie_and_TV_GT5.csv',dtype={'user':object,'item':object,
#         'rating':np.int64,'template':object,'predicted':object,
#         'hist_item':object,'unixReviewTime':np.int64,'hist_review':object,
#         'neg_sample':object,'item_index':np.int64,'hist_item':object,'neg_sample_item':object})

# # print(peter_data.columns,peter_data.dtypes)

# peter_data.to_pickle('./FMLP/data/Movie_and_TV_GT5_new.pickle')
