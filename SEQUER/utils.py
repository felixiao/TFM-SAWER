import os
import math
import torch
import heapq
import random
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from transformers import GPT2Tokenizer

from constants import *
from rouge import rouge
from bleu import compute_bleu
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset

def init_seed(seed=RNG_SEED, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb
    log_info(f'feature_coverage_ratio: len feature_batch={len(feature_batch)}, len feature_set = {len(feature_set)}',level=LOG_DEBUG_DETAIL)
    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    log_info(f'Preditct Len={len(predicted)}, max:{max_r}, min:{min_r}',level=LOG_DEBUG_DETAIL)
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


class WordDictionary:
    def __init__(self, initial_tokens=None):
        self.idx2word = []
        if initial_tokens is not None:
            self.idx2word += initial_tokens
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.__word2count = {}

    def add_sentence(self, sentence):
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] += 1

    def __len__(self):
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size=20000):
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}


class EntityDictionary:
    def __init__(self, initial_tokens=None):
        self.idx2entity = []
        if initial_tokens is not None:
            self.idx2entity += initial_tokens
        self.entity2idx = {e: i for i, e in enumerate(self.idx2entity)}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class SEQUER_DataLoader:
    def __init__(self, data_path, index_dir, vocab_size, tokenizer, word_len, seq_mode=0, test_flag=False,load_direct=False,max_hist_len=20):
        initial_tokens = [BOS_TOK, EOS_TOK, PAD_TOK, UNK_TOK]
        self.max_hist_len = max_hist_len
        self.word_dict = WordDictionary(initial_tokens)
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary(initial_tokens)
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        if load_direct:
            self.initialize_direct(data_path)
        else:
            self.initialize(data_path, seq_mode)
            
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx[UNK_TOK]
        self.feature_set = set()
        self.tokenizer = tokenizer
        self.seq_len = word_len

        if load_direct:
            self.feature_set, self.train, self.valid, self.test ,self.user2feature, self.item2feature = self.loaddirectly(data_path)
        else:
            self.train, self.valid, self.test ,self.user2feature, self.item2feature = self.load_data(data_path, index_dir, seq_mode, test_flag,save=True)

        self.nuser = len(self.user_dict)
        self.nitem = len(self.item_dict)
        if tokenizer:
            self.ntokens = len(tokenizer)
        else:
            self.ntokens = len(self.word_dict)
        

    def initialize(self, data_path, seq_mode):
        assert os.path.exists(os.path.join(data_path,'reviews_new.pickle'))
        reviews = pickle.load(open(os.path.join(data_path,'reviews_new.pickle'), 'rb'))
        log_info(f'seq_mode:{seq_mode}',gpu_id=int(os.environ["LOCAL_RANK"]),level=LOG_DEBUG_DETAIL)
        log_info(f'reviews {reviews.index}',gpu_id=int(os.environ['LOCAL_RANK']))

        for r in tqdm(reviews,desc='Init',ncols=100,position=0, leave=True):
        # for review in reviews:
            self.user_dict.add_entity(r[U_COL])
            self.item_dict.add_entity(r[I_COL])
            # self.user_dict.add_entity(review[U_COL])
            # self.item_dict.add_entity(review[I_COL])
            # template = review['template']
            # (fea, adj, tem, sco) = template
            self.word_dict.add_sentence(r[REV_COL])
            self.word_dict.add_word(r[FEAT_COL])
            if seq_mode >= HIST_REV_MODE: # FMLPETER mode = 2, HIST_REV_MODE = 3; 
                for j in range(len(r[HIST_I_COL])):
                    # (fea, adj, tem, sco) = template
                    self.word_dict.add_sentence(r[HIST_REV_COL][j])
                    self.word_dict.add_word(r[HIST_FEAT_COL][j])
            rating = r[RAT_COL]
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating
        log_info(f'user dict count:{len(self.user_dict)}\nitem dict count:{len(self.item_dict)}\nword dict count:{len(self.word_dict)}\nmax rating:{self.max_rating}\nmin rating:{self.min_rating}',gpu_id=int(os.environ["LOCAL_RANK"]))

        with open(os.path.join(data_path,'init.pickle'), 'wb') as handle:
            pickle.dump({'user':self.user_dict,'item':self.item_dict,'word':self.word_dict,'max_rating':self.max_rating,'min_rating':self.min_rating}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def initialize_direct(self,data_path):
        log_info('Init Start',gpu_id = int(os.environ["LOCAL_RANK"]))
        with open(os.path.join(data_path,'init.pickle'), 'rb') as handle:
            init=pickle.load(handle)
            self.user_dict = init['user']
            self.item_dict = init['item']
            self.word_dict = init['word']
            self.max_rating = init['max_rating']
            self.min_rating = init['min_rating']
        log_info('Init End',gpu_id = int(os.environ["LOCAL_RANK"]))

    def load_data(self, data_path, index_dir, seq_mode=0, test_flag=False,save=True):
        data = []
        merge_str = f' {EOS_TOK} {BOS_TOK} '
        reviews = pickle.load(open(os.path.join(data_path,'reviews_new.pickle'), 'rb'))

        for r in tqdm(reviews,desc='Load',ncols=100,position=0, leave=True):
        # for review in reviews:
            # (fea, adj, tem, sco) = review['template']
            data.append({U_COL: self.user_dict.entity2idx[r[U_COL]],
                         I_COL: self.item_dict.entity2idx[r[I_COL]],
                         RAT_COL: r[RAT_COL],
                         REV_COL: self.seq2ids(r[REV_COL]),
                         FEAT_COL: self.word_dict.word2idx.get(r[FEAT_COL], self.__unk)})
            if seq_mode >= HIST_I_MODE:# FMLPETER mode = 2, HIST_I_MODE = 1; 
                # fix make it comp use flag or other
                # data[i][HIST_I_COL] = reviews[HIST_I_IDX_COL][i] # hist_item_index
                data[-1][HIST_I_COL] = [self.item_dict.entity2idx[id] for id in r[HIST_I_COL]]
                data[-1][HIST_RAT_COL] = r[HIST_RAT_COL]
                if seq_mode >= HIST_REV_MODE: # FMLPETER mode = 2, HIST_REV_MODE = 3; 
                    # TODO: Try with EOS token to transfer info from one review to another
                    data[-1][HIST_REV_COL] = self.seq2ids(merge_str.join(r[HIST_REV_COL]))
            if r[FEAT_COL] in self.word_dict.word2idx:
                self.feature_set.add(r[FEAT_COL])
            else:
                self.feature_set.add(UNK_TOK)

        train_index, valid_index, test_index = self.load_index(index_dir)
        if test_flag:
            train_index = train_index[:1000]
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        for idx in train_index:
            review = data[idx]
            train.append(review)
            u = review[U_COL]
            i = review[I_COL]
            f = review[FEAT_COL]
            if u in user2feature:
                user2feature[u].append(f)
            else:
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)
            else:
                item2feature[i] = [f]
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        
        if save:
            with open(os.path.join(data_path,'featureset.pickle'), 'wb') as handle:
                pickle.dump(self.feature_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(data_path,'train.pickle'), 'wb') as handle:
                pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(data_path,'valid.pickle'), 'wb') as handle:
                pickle.dump(valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(data_path,'test.pickle'), 'wb') as handle:
                pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(data_path,'user2feat.pickle'),'wb') as handle:
                pickle.dump(user2feature, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(data_path,'item2feat.pickle'),'wb') as handle:
                pickle.dump(item2feature, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return train, valid, test,user2feature, item2feature

    def loaddirectly(self,data_path):
        log_info('Load Start',gpu_id = int(os.environ["LOCAL_RANK"]))
        feature_set = pickle.load(open(os.path.join(data_path,'featureset.pickle'), 'rb'))
        
        train = pickle.load(open(os.path.join(data_path,'train.pickle'), 'rb'))
        log_info(f'train len:{len(train)}',gpu_id = int(os.environ["LOCAL_RANK"]))
        
        valid = pickle.load(open(os.path.join(data_path,'valid.pickle'), 'rb'))
        test = pickle.load(open(os.path.join(data_path,'test.pickle'), 'rb'))
        user2feature = pickle.load(open(os.path.join(data_path,'user2feat.pickle'), 'rb'))
        item2feature = pickle.load(open(os.path.join(data_path,'item2feat.pickle'), 'rb'))
        
        log_info('Load End',gpu_id = int(os.environ["LOCAL_RANK"]))
        
        return feature_set, train, valid, test, user2feature, item2feature

    def seq2ids(self, seq):
        if self.tokenizer:
            tokens = self.tokenizer(seq)['input_ids']
            text = self.tokenizer.decode(tokens[:self.seq_len])
            return text
        else:
            return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        train_index = np.genfromtxt(os.path.join(index_dir, 'train.index'), delimiter=' ', dtype=int)
        valid_index = np.genfromtxt(os.path.join(index_dir, 'validation.index'), delimiter=' ', dtype=int)
        test_index = np.genfromtxt(os.path.join(index_dir, 'test.index'), delimiter=' ', dtype=int)
        return train_index, valid_index, test_index

def sentence_format(sentence, max_len, pad, bos, eos, prefix=None):
    if prefix is not None:
        sentence = [bos] + prefix + [eos] + sentence
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_len - length)

def get_review_index(reviews, eos):
    ixs = [0]
    for w in reviews[1:]:
        if w == eos:
            ixs.append(ixs[-1] + 1)
        else:
            ixs.append(ixs[-1])

def now_time():
    return f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def ids2tokens(ids, word2idx, idx2word):
    eos = word2idx[EOS_TOK]
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens

def plot_mask(mask,filename,label='',title='Attention Mask',xticks='',yticks=''):
    plt.figure(figsize=(15, 15))
    revmask = (mask==False)
    
    ax = sns.heatmap(revmask,cbar=False,linewidths=0.1,linecolor='red',square=True,
        # xticklabels=np.arange(1, revmask.shape[1]+1, 1),
        xticklabels=xticks,
        yticklabels=yticks)
    ax.set(xlabel=label+'\nSource', ylabel="Target")
    ax.xaxis.tick_top()
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    log_info(f'Save Mask to {filename}',gpu_id=int(os.environ['LOCAL_RANK']))

def save_results(curr_res):
    if curr_res['GPU_ID'] ==0:
        results = pd.DataFrame.from_records([curr_res],columns=COLUMN_NAME)
        if os.path.exists(os.path.join(RES_PATH, 'results.csv')):
            df_old = pd.read_csv(os.path.join(RES_PATH, 'results.csv'),index_col=None)
            results = pd.concat([df_old,results], ignore_index=True,sort=False)
        results = results.astype({'Split_ID': 'int32','Seed':'int32',
                                'HistLen':'int32','TextLen':'int32',
                                'Batch_size':'int32','Total_Epoch':'int32',
                                'Num_GPU':'int32','GPU_ID':'int32'})
        results.to_csv(os.path.join(RES_PATH, 'results.csv'), index=False,columns=COLUMN_NAME)
        log_info(f'Saved result to {os.path.join(RES_PATH, "results.csv")}',gpu_id=int(os.environ['LOCAL_RANK']))

class AmazonDataset(Dataset):
    def __init__(self,data,tokenizer, word2idx, seq_mode=0, hist_len=20,word_len=15):
        super(Dataset, self).__init__()
        bos = word2idx[BOS_TOK]
        eos = word2idx[EOS_TOK]
        pad = word2idx[PAD_TOK]
        unk = word2idx[UNK_TOK]

        u, i, r, t, f= [], [], [], [], []
        self.len = len(data)
        for x in data:
            u.append(x[U_COL])
            if seq_mode >= HIST_I_MODE:
                i.append([unk] * (hist_len - len(x[HIST_I_COL])) + x[HIST_I_COL][-hist_len:])
                # i.append([unk] * (HIST_LEN - len(x[HIST_I_COL])) + x[HIST_I_COL][-HIST_LEN:] + [x[I_COL]])
                if seq_mode == HIST_I_MODE + 1:
                    r.append(x[RAT_COL])
                else:
                    r.append([pad] * (hist_len - len(x[HIST_I_COL])) + x[HIST_RAT_COL][-hist_len:] + [x[RAT_COL]])
            else:
                i.append(x[I_COL])
                r.append(x[RAT_COL])
            if tokenizer:
                t.append('{} {} {}'.format(bos, x[REV_COL], eos))
            else:
                if seq_mode >= HIST_REV_MODE:
                    t.append(sentence_format(x[REV_COL], word_len * (hist_len + 1), pad, bos, eos, x[HIST_REV_COL]))
                else:
                    t.append(sentence_format(x[REV_COL], word_len, pad, bos, eos))
            f.append([x[FEAT_COL]])
            
        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.feature = torch.tensor(f, dtype=torch.int64).contiguous()
        
        # PEPLER
        if tokenizer:
            log_info(f't={type(t)}',gpu_id=int(os.environ["LOCAL_RANK"]))
            encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
            self.seq = encoded_inputs['input_ids'].contiguous()
            self.mask = encoded_inputs['attention_mask'].contiguous()
        else:
            self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
            self.mask = None
        
    def __len__(self):
        return self.len
    
    def __getitem__(self,idx):
        if self.mask:
            return self.user[idx], self.item[idx], self.rating[idx], self.seq[idx], self.feature[idx],self.mask[idx]
        else:
            return self.user[idx], self.item[idx], self.rating[idx], self.seq[idx], self.feature[idx]
        
def load_data(data_path, index_dir, seq_mode,batch_size, vocab_size,hist_len,word_len,test_flg=False,load_direct=False,pre_trained='GPT2'):
    log_info(f'Loading data Fold {index_dir}',gpu_id=int(os.environ["LOCAL_RANK"]))

    if pre_trained == 'GPT2': # PEPLER
        tokenizer = GPT2Tokenizer(vocab_file='PEPLER/GPT2/vocab.json',merges_file='PEPLER/GPT2/merges.txt',bos_token=BOS_TOK, eos_token=EOS_TOK, pad_token=PAD_TOK)
    else: # Other
        tokenizer = None

    corpus = SEQUER_DataLoader(data_path, index_dir, vocab_size,tokenizer, word_len, seq_mode, test_flg,load_direct,hist_len)
    
    word2idx = corpus.word_dict.word2idx
    idx2word = corpus.word_dict.idx2word
    feature_set = corpus.feature_set

    train_data = AmazonDataset(corpus.train,tokenizer,word2idx,seq_mode,hist_len,word_len)
    val_data = AmazonDataset(corpus.valid,tokenizer,word2idx,seq_mode,hist_len,word_len)
    test_data = AmazonDataset(corpus.test,tokenizer,word2idx,seq_mode,hist_len,word_len)

    train_dataloader = DataLoader(train_data,
                                    batch_size=batch_size, 
                                    pin_memory=True,
                                    shuffle=False,
                                    sampler=DistributedSampler(train_data))
    val_dataloader = DataLoader(val_data,
                                    batch_size=batch_size,
                                    pin_memory=True,
                                    shuffle=False,
                                    sampler=DistributedSampler(val_data))
    test_dataloader = DataLoader(test_data,
                                    batch_size=batch_size,
                                    pin_memory=True,
                                    shuffle=False)

    return {'corpus':corpus,'word2idx':word2idx,'idx2word':idx2word, 'feature_set':feature_set,
            'train_data':train_data,'val_data':val_data,'test_data':test_data,
            'train_dataloader':train_dataloader, 'val_dataloader':val_dataloader,'test_dataloader':test_dataloader}

def log_info(info,level=LOG_INFO,gpu_id=-1,time=True,gpu=True):
    if level<=LOG_LEVEL and gpu_id <= 0:
        gpustr = f' GPU {gpu_id}' if gpu_id>=0 and gpu else ''
        prefix = f'[{now_time()}{gpustr}]: ' if time else ''
        
        print(prefix+info)

class TrainHistory():
    def __init__(self,model_name) -> None:
        self.model_name = model_name
        self.epoch = 0
        self.train_context_loss = []
        self.train_text_loss = []
        self.train_rating_loss = []
        self.train_loss = []

        self.val_context_loss = []
        self.val_text_loss = []
        self.val_rating_loss = []
        self.val_loss = []
    
    def __len__(self):
        return self.epoch+1

    def __str__(self) -> str:
        return f'{self.model_name}: Ep {self.epoch} Len {len(self.train_loss)}'

    def add_history(self,epoch,hist_dict):
        if self.epoch>=epoch:
            self.train_context_loss = self.train_context_loss[:epoch-1]
            self.train_text_loss = self.train_text_loss[:epoch-1]
            self.train_rating_loss= self.train_rating_loss[:epoch-1]
            self.train_loss = self.train_loss[:epoch-1]

            self.val_context_loss = self.val_context_loss[:epoch-1]
            self.val_text_loss = self.val_text_loss[:epoch-1]
            self.val_rating_loss = self.val_rating_loss[:epoch-1]
            self.val_loss =self.val_loss[:epoch-1]
        self.epoch = epoch
        self.train_context_loss.append(hist_dict['train_context_loss'])
        self.train_text_loss.append(hist_dict['train_text_loss'])
        self.train_rating_loss.append(hist_dict['train_rating_loss'])
        self.train_loss.append(hist_dict['train_loss'])

        self.val_context_loss.append(hist_dict['val_context_loss'])
        self.val_text_loss.append(hist_dict['val_text_loss'])
        self.val_rating_loss.append(hist_dict['val_rating_loss'])
        self.val_loss.append(hist_dict['val_loss'])
    
    def get_dict(self):
        return {'epoch':np.arange(1,len(self.train_loss)+1,1),
                'train_context_loss':self.train_context_loss,
                'train_text_loss':  self.train_text_loss,
                'train_rating_loss':self.train_rating_loss,
                'train_loss':       self.train_loss,
                'val_context_loss': self.val_context_loss,
                'val_text_loss':    self.val_text_loss,
                'val_rating_loss':  self.val_rating_loss,
                'val_loss':         self.val_loss}

    def save_plot(self):

        dim = np.arange(1,len(self.train_loss)+1,1)
        df_hist = pd.DataFrame(self.get_dict(),index=dim)
        df_hist.to_csv(os.path.join(RES_PATH,f'history-{self.model_name}.csv'),index=None)

        plt.plot(dim,self.train_loss,label='train_loss')
        plt.plot(dim,self.val_loss,label='val_loss')
        plt.title(f'{self.model_name} Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(RES_PATH,f'history-{self.model_name}.jpg'))
        plt.close()
