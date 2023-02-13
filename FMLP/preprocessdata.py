### Convert IDs of User, item

import pandas as pd
from tqdm import tqdm,trange
import random
import numpy as np
import os

def check_path(path):
  if not os.path.exists(path):
      os.makedirs(path)
      print(f'{path} created')

class PreprocessData():
  def __init__(self,data_file='data/reviews_new.pickle',data_name = 'Movie_and_TV',index_dir='data'):
    self.data_file = data_file
    self.data_name = data_name
    self.index_dir = index_dir
    check_path(index_dir)
    check_path(os.path.join(index_dir,'0'))

  def load_file(self):
    self.peter_data = pd.DataFrame.from_records(pd.read_pickle(self.data_file))

    print(list(self.peter_data.columns))
    unique_users = list(self.peter_data['user'].unique())
    unique_items = set(self.peter_data['item'].unique())
    print('Unique/Total User:',len(unique_users),'/',len(self.peter_data['user']))
    print('Unique/Total Item:',len(unique_items),'/',len(self.peter_data['item']))
    self.item_dict = {i:id for id, i in enumerate(unique_items)}
    self.user_dict = {i:id for id, i in enumerate(unique_users)}

    del unique_users
    del unique_items

  def convert_index(self):
    self.peter_data['user_index'] = self.peter_data.apply(lambda row: self.user_dict[row['user']],axis=1)
    self.peter_data['item_index'] = self.peter_data.apply(lambda row: self.item_dict[row['item']],axis=1)
    self.peter_data['hist_item_index'] = self.peter_data.apply(lambda row:self.hist_item_to_index(row,self.item_dict),axis=1)
    self.Create_Neg_Samples()

  def create_neg_samples(self):
    self.fmlp_data = self.peter_data[['user_index','item_index','hist_item_index']].to_numpy()

    # print(fmlp_data.shape)
    # print(fmlp_data[:5])

    item_list = np.unique(self.fmlp_data[:,1])
    user_list = np.unique(self.fmlp_data[:,0])
    neg_samples_index = []
    neg_item_index = []
    user_itemset = {}

    for u in tqdm(user_list,desc='User_Item'):
      user_items = self.fmlp_data[self.fmlp_data[:,0]==user_list[u]][:,2]
      user_itemset[u] = np.unique(np.concatenate(user_items))

    item_set = set(item_list)

    for u in tqdm(self.fmlp_data[:,0],desc='Neg Samples'):
      list_neg = list(item_set.difference(set(user_itemset[u])))
      neg_item_index.append(random.sample(list_neg,1))
      neg_samples_index.append(random.sample(list_neg,99))

    self.fmlp_data = np.concatenate([self.fmlp_data, neg_item_index, neg_samples_index], axis=1)
    self.peter_data['neg_samples_index'] = neg_samples_index
    self.peter_data['neg_item_index'] = neg_item_index
    np.save(os.path.join(self.index_dir,self.data_name+'_index'),self.fmlp_data)
    self.peter_data.to_pickle(os.path.join(self.index_dir,self.data_name+'_index.pickle'))
    del item_list
    del user_list
    del neg_samples_index
    del neg_item_index
    del user_itemset


  def select_data_num_history(self,limit=5):
    self.peter_data['len_hist_item'] = self.peter_data.apply(lambda row:len(row['hist_item_index']),axis=1)
    self.peter_data = self.peter_data[self.peter_data['len_hist_item'] > limit]
    self.peter_data.drop(['len_hist_item'],axis=1,inplace=True)
    self.peter_data.to_pickle(os.path.join(self.index_dir,f'{self.data_name}_GT{limit}.pickle'))
    print(f'saved {self.data_name}_GT{limit} pickle')

    fmlp_data_len = [[len(self.fmlp_data[i,2])] for i in range(len(self.fmlp_data))]
    fmlp_data_gt= np.concatenate([fmlp_data_len,self.fmlp_data], axis=1)
    fmlp_data_gt = fmlp_data_gt[np.where(fmlp_data_gt[:,0]>limit)]
    fmlp_data_gt = np.delete(fmlp_data_gt,0,axis=1)
    # print(fmlp_data_gt[:5])
    np.save(os.path.join(self.index_dir,f'{self.data_name}_GT{limit}'),fmlp_data_gt)
    print(f'saved {self.data_name}_GT{limit} npy')
    del fmlp_data_gt
    del fmlp_data_len

  def split_data(self,test_ratio=0.1):
    val_ratio = test_ratio

    self.peter_data.sort_values(by=['user', 'timestamp'], inplace=True)
    ucounts = self.peter_data['user'].value_counts().values
    uoffsets = ucounts.cumsum()
    split_ixs = np.zeros((self.peter_data.shape[0], ), dtype=int)
    if isinstance(test_ratio, float):
        assert isinstance(val_ratio, float)
        assert test_ratio < 1.0
        tst_start_ixs = uoffsets - (ucounts * test_ratio).astype(int)
        val_start_ixs = tst_start_ixs - (ucounts * val_ratio).astype(int)
    elif isinstance(test_ratio, int):
        assert isinstance(val_ratio, int)
        assert all(ucounts > (test_ratio + val_ratio))
        tst_start_ixs = uoffsets - test_ratio
        val_start_ixs = tst_start_ixs - val_ratio
    else:
        raise TypeError('test_ratio is neither int nor float')
    for vix, tix, offset in zip(val_start_ixs, tst_start_ixs, uoffsets):
        split_ixs[tix:offset] = 2
        split_ixs[vix:tix] = 1

    np.savetxt(os.path.join(self.index_dir, '0', 'train.index'),
               self.peter_data.index.values[split_ixs == 0],
               delimiter=' ', fmt="%d")

    np.savetxt(os.path.join(self.index_dir, '0', 'validation.index'),
               self.peter_data.index.values[split_ixs == 1],
               delimiter=' ', fmt="%d")

    np.savetxt(os.path.join(self.index_dir, '0', 'test.index'),
               self.peter_data.index.values[split_ixs == 2],
               delimiter=' ', fmt="%d")

    print('Finished!')

  def hist_item_to_index(self,row,item_dict):
    histitem_list = [item_dict[i] for i in row['hist_item']]
    histitem_list.append(row['item_index'])
    return histitem_list