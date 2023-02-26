import os
import math
import torch
import heapq
import random
import pickle
import datetime
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tokenizers import Tokenizer, trainers
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .constants import *
from .rouge import rouge
from .bleu import compute_bleu


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


def feature_matching_ratio(feature_batch, test_feature, ignore=None):
    count = 0
    norm = sum([f != ignore for f in test_feature])
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea != ignore and fea in fea_set:
            count += 1

    return count / norm  # len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

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


class EntityTokenizer:
    def __init__(self, special_tokens=None):
        self.idx2entity = []
        if special_tokens is not None:
            self.idx2entity += list(special_tokens.values())
        self.entity2idx = pd.Series({e: i for i, e in enumerate(self.idx2entity)})

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def add_entities(self, es: list):
        offset = len(self.entity2idx)
        self.entity2idx = pd.concat([self.entity2idx, pd.Series({es: i + offset for i, es in enumerate(es)})])
        self.idx2entity.extend(es)

    def encode(self, entities):
        return self.entity2idx[entities].values

    def __len__(self):
        return len(self.idx2entity)


def get_context(docs, pos_tags=('NOUN', 'ADJ')):
    nlp = spacy.load("en_core_web_sm")
    contexts = []
    for s in nlp.pipe(docs):
        contexts.append(' '.join([tok for tok in s if tok.pos_ in pos_tags]))
    return contexts


def get_tokenizer(dataset, data, tokenizer_name='default', special_tokens=None, vocab_size=VOCAB_SIZE, retrain=False,
                  padding_side='right', truncation_side='right'):
    tok_path = os.path.join(DATA_PATHS[dataset], f'{tokenizer_name}.json')
    special_tok_vals = list(special_tokens.values())

    if tokenizer_name in ['gpt2']:
        tokenizer = Tokenizer.from_pretrained(tokenizer_name, bos_token=BOS_TOK, eos_token=EOS_TOK, pad_token=PAD_TOK)
    elif os.path.exists(tok_path) and not retrain:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path, padding_side=padding_side,
                                            truncation_side=truncation_side)
        tokenizer.add_special_tokens(special_tokens)
    elif tokenizer_name == 'item':
        tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOK))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        trainer = trainers.WordLevelTrainer(special_tokens=special_tok_vals)
        tokenizer.train_from_iterator(iter(data), trainer=trainer, length=len(data))
        tokenizer.save(tok_path)
        tokenizer = get_tokenizer(dataset, data, tokenizer_name, padding_side='left', truncation_side='left')
    elif tokenizer_name == 'default':
        tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOK))
        # tokenizer.normalizer = Lowercase()
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS_TOK} $0 {EOS_TOK}",
            special_tokens=[(BOS_TOK, special_tok_vals.index(BOS_TOK)), (EOS_TOK, special_tok_vals.index(EOS_TOK))],
        )
        trainer = trainers.WordLevelTrainer(special_tokens=special_tok_vals, show_progress=True,
                                            vocab_size=vocab_size + len(special_tokens))
        tokenizer.train_from_iterator(iter(data), trainer=trainer, length=len(data))
        tokenizer.save(tok_path)
    else:
        raise ValueError(f'Tokenizer not implemented: {tokenizer_name}')

    return tokenizer


def encode_hrev(reviews, tokenizer, truncation=False, max_len=None):
    enc_r = tokenizer.batch_encode_plus(reviews, truncation=truncation, max_length=max_len, return_length=True,
                                        add_special_tokens=False)
    segment_ids = [i for i, r in enumerate(enc_r) for _ in r]
    return segment_ids, enc_r


class DataLoader:
    def __init__(self, dataset, fold, vocab_size, seq_mode=0, tokenizer=None, test_flag=False, mod_context_flag=False):
        # initial_tokens = [BOS_TOK, EOS_TOK, PAD_TOK, UNK_TOK, SEP_TOK]
        # self.word_dict = WordDictionary(initial_tokens)
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(dataset, fold, tokenizer, vocab_size, seq_mode, test_flag, mod_context_flag)
        # self.word_dict.keep_most_frequent(vocab_size)
        # self.__unk = self.word_dict.word2idx[UNK_TOK]
        # self.feature_set = set()
        # self.tokenizer = tokenizer
        # self.train, self.valid, self.test = self.load_data(data_path, index_dir, seq_mode, test_flag)

    def initialize(self, dataset, fold, tokenizer, vocab_size, seq_mode, test_flag, mod_context_flag):
        special_tokens = {
            'pad_token': PAD_TOK,
            'bos_token': BOS_TOK,
            'eos_token': EOS_TOK,
            'unk_token': UNK_TOK,
            'sep_token': SEP_TOK
        }

        index_dir = os.path.join(DATA_PATHS[dataset], str(fold))
        data_path = os.path.join(DATA_PATHS[dataset], 'reviews_new.pickle')

        assert os.path.exists(data_path)
        reviews = pd.DataFrame.from_records(pickle.load(open(data_path, 'rb')))
        train_index, valid_index, test_index = self.load_index(index_dir)
        if test_flag:
            train_index = train_index[:1000]

        if seq_mode >= HIST_I_MODE:
            reviews[I_COL] = reviews[I_COL].values[:, None].tolist()
            reviews[I_COL] = (reviews[HIST_I_COL] + reviews[I_COL]).str.join(' ')
        reviews.drop(HIST_I_COL, axis=1, inplace=True)

        if mod_context_flag:
            reviews[CONTEXT_COL] = get_context(reviews[REV_COL].values.tolist(), pos_tags=('NOUN', 'ADJ'))

        self.train = reviews.loc[train_index].reset_index(drop=True)
        self.valid = reviews.loc[valid_index].reset_index(drop=True)
        self.test = reviews.loc[test_index].reset_index(drop=True)

        # reviews = reviews.loc[np.concatenate([train_index, valid_index, test_index])].reset_index(drop=True)

        # Original code: train_reviews = reviews[REV_COL].values.tolist()
        train_reviews = self.train[REV_COL].values.tolist() + self.valid[REV_COL].values.tolist()
        self.word_tok = get_tokenizer(dataset, train_reviews, tokenizer, special_tokens, vocab_size)
        self.user_tok = EntityTokenizer()
        self.user_tok.add_entities(reviews[U_COL].unique().tolist())
        self.max_rating = reviews[RAT_COL].max()
        self.min_rating = reviews[RAT_COL].min()

        train_items = self.train[I_COL].values.tolist() + self.valid[I_COL].values.tolist()
        self.item_tok = get_tokenizer(dataset, train_items, 'item', special_tokens)

        # Assert all test items appear in the training set
        # assert not set(self.test[I_COL].explode().unique()).difference(set([i for si in train_items for i in si]))
        for df in [self.train, self.valid, self.test]:
            df[U_COL] = self.user_tok.encode(df[U_COL].values)
            enc_i = self.item_tok.batch_encode_plus(df[I_COL].values, truncation=True, max_length=HIST_LEN+1,
                                                    return_length=True)
            df[I_COL] = self.item_tok.pad(enc_i)['input_ids']

            # Add item segment ids (with padding - padding belongs to segment id 0)
            max_l = max(enc_i['length'])
            if self.item_tok.padding_side == 'left':
                df[SEG_I_COL] = [[0] * (max_l - l) + list(range(l)) for l in enc_i['length']]
            else:
                df[SEG_I_COL] = [list(range(l)) + [0] * (max_l - l) for l in enc_i['length']]

            # Encode and truncate/pad the candidate item's review (always same truncation for comp. purposes)
            enc_r = self.word_tok.batch_encode_plus(df[REV_COL].values, padding=True, truncation=True,
                                                    max_length=TXT_LEN + 2, return_tensors='np')
            df[REV_COL] = enc_r['input_ids'].tolist()
            df[SEG_REV_COL] = np.broadcast_to(np.array(df[SEG_I_COL].tolist())[:, -1:], (df.shape[0], TXT_LEN + 2))
            if seq_mode >= HIST_REV_MODE:
                # Get segment ids for historical reviews (with padding)
                segment_ids, enc_reviews = df[HIST_REV_COL].apply(lambda r: encode_hrev(r, self.word_tok,
                                                                                        truncation=True,
                                                                                        max_len=TXT_LEN),
                                                                  axis=1)
            df[REV_COL] = self.word_tok.encode_batch(df[U_COL].values)

        # for review in reviews:
        #     self.user_dict.add_entity(review[U_COL])
        #     self.item_dict.add_entity(review[I_COL])
        #     self.word_dict.add_sentence(review[REV_COL])
        #     # # NOTE: I've added the next line of code so that all words are in dictionary. Comment it out if necessary
        #     if review[FEAT_COL] != '':
        #         self.word_dict.add_word(review[FEAT_COL])
        #     rating = review[RAT_COL]
        #     if self.max_rating < rating:
        #         self.max_rating = rating
        #     if self.min_rating > rating:
        #         self.min_rating = rating

    def load_data(self, data_path, index_dir, seq_mode=0, test_flag=False):
        data = []
        merge_str = f' {EOS_TOK} {BOS_TOK} '
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            # (fea, adj, tem, sco) = review['template']
            data.append({U_COL: self.user_dict.entity2idx[review[U_COL]],
                         I_COL: self.item_dict.entity2idx[review[I_COL]],
                         RAT_COL: review[RAT_COL],
                         REV_COL: self.seq2ids(review[REV_COL]),
                         # NOTE: This line is different in NRT/Att2Seq and PETER
                         FEAT_COL: self.word_dict.word2idx.get(review[FEAT_COL], self.__unk)})
            if seq_mode >= HIST_I_MODE:
                data[-1][HIST_I_COL] = [self.item_dict.entity2idx[i] for i in review[HIST_I_COL]]
                data[-1][HIST_RAT_COL] = review[HIST_RAT_COL]
                if seq_mode >= HIST_REV_MODE:
                    # TODO: Try with EOS token to transfer info from one review to another
                    data[-1][HIST_REV_COL] = self.seq2ids(merge_str.join(review[HIST_REV_COL]))
            # NOTE: This if-statement is not present in NRT/Att2Seq code
            if review[FEAT_COL] in self.word_dict.word2idx:
                self.feature_set.add(review[FEAT_COL])
            # else:
            #     self.feature_set.add(UNK_TOK)

        train_index, valid_index, test_index = self.load_index(index_dir)
        if test_flag:
            train_index = train_index[:1000]
        train, valid, test = [], [], []
        for idx in train_index:
            train.append(data[idx])
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test

    def seq2ids(self, seq):
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


class Batchify:
    def __init__(self, data, word2idx, batch_size=128, seq_mode=0, shuffle=False):
        bos = word2idx[BOS_TOK]
        eos = word2idx[EOS_TOK]
        pad = word2idx[PAD_TOK]
        unk = word2idx[UNK_TOK]

        u, i, r, t, tix, f = [], [], [], [], [], []
        for x in data:
            u.append(x[U_COL])
            if seq_mode >= HIST_I_MODE:
                i.append([unk] * (HIST_LEN - len(x[HIST_I_COL])) + x[HIST_I_COL][-HIST_LEN:] + [x[I_COL]])
                if seq_mode == HIST_I_MODE + 1:
                    r.append(x[RAT_COL])
                else:
                    r.append([pad] * (HIST_LEN - len(x[HIST_I_COL])) + x[HIST_RAT_COL][-HIST_LEN:] + [x[RAT_COL]])
            else:
                i.append(x[I_COL])
                r.append(x[RAT_COL])
            if seq_mode >= HIST_REV_MODE:
                t.append(sentence_format(x[REV_COL], TXT_LEN * (HIST_LEN + 1), pad, bos, eos, x[HIST_REV_COL]))
            else:
                t.append(sentence_format(x[REV_COL], TXT_LEN, pad, bos, eos))
            f.append([x[FEAT_COL]])

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.feature = torch.tensor(f, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                # Random seed was not fixed in the original code
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]  # (batch_size, seq_len)
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        feature = self.feature[index]  # (batch_size, 1)
        return user, item, rating, seq, feature


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def ids2tokens(ids, word2idx, idx2word):
    # TODO: Check the fact that this function changes in PEPLER and uses post-processing
    eos = word2idx[EOS_TOK]
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens


def plot_mask(mask):
    plt.imshow(mask, cmap='Greys', interpolation='nearest')
    plt.xticks(np.arange(0, mask.shape[1], 1))
    plt.yticks(np.arange(0, mask.shape[0], 1))
    plt.grid()
    plt.show()


def save_results(curr_res):
    is_new_file = not os.path.isfile(os.path.join(RES_PATH, 'results.csv'))
    if not is_new_file:
        results = pd.read_csv(os.path.join(RES_PATH, 'results.csv'))
    else:
        results = pd.DataFrame(columns=curr_res.keys())
    missing_cols = set(curr_res.keys()).difference(results.columns)
    for c in missing_cols:
        curr_res[c] = 0
    results = results.append(pd.DataFrame().from_records([curr_res]))
    results.to_csv(os.path.join(RES_PATH, 'results.csv'), index=False)
