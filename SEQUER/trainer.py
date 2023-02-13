import math
import torch
import pandas as pd
import logging
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from modules import PETER, SEQUER, Att2Seq, FMLPETER, PEPLER, SAWER
from utils import rouge_score, bleu_score, SEQUER_DataLoader, now_time, ids2tokens, unique_sentence_percent, \
    root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, \
    feature_diversity, save_results,AmazonDataset, log_info, TrainHistory


from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from torch.nn.parallel import DistributedDataParallel as DDP

from constants import *

class Trainer:
    def __init__(self, dataset, fold, prediction_path, model_name, data_helper, cfg, device, test_flg=False,
                 gen_flg=False,gpu_id=0,num_gpu=1):
        self.model_name = model_name
        # log_info(f'NUMEXPR_MAX_THREADS: { ["NUMEXPR_MAX_THREADS"]}',gpu_id=gpu_id)
        # self.gpu_id = int(os.environ["LOCAL_RANK"])
        gpu_id = int(os.environ["LOCAL_RANK"])
        # log_info(f'LOCAL RANK: {os.environ["LOCAL_RANK"]}',gpu_id=gpu_id)
        self.gpu_id = gpu_id
        self.use_feature = cfg.get('use_feature', True)
        self.out_words = cfg.get('text_len',15)
        self.text_reg = cfg.get('text_reg', 0)
        self.context_reg = cfg.get('context_reg', 0)
        self.rating_reg = cfg.get('rating_reg', 0)
        self.item_reg = cfg.get('item_reg', 0)
        log_info(f'text_reg:{self.text_reg}, context_reg:{self.context_reg}, rating_reg:{self.rating_reg}, item_reg:{self.item_reg}',gpu_id=gpu_id)
        self.clip_norm = cfg.get('clip_norm', 1.0)
        self.epochs = cfg.get('epochs', 100)
        self.endure_times = cfg.get('endure_times', 5)
        self.log_interval = cfg.get('log_interval',4)
        self.gen_flg = gen_flg
        self.seq_mode = cfg.get('seq_mode', 0)
        self.batch_size = cfg.get('batch_size', 128)
        self.vocab_size = cfg.get('vocab_size', 5000)
        self.hist_len = int(cfg.get('hist_len',0))
        self.user_len = int(cfg.get('user_len',1))
        self.item_len = int(cfg.get('item_len',1))
        self.warmup_epoch = int(cfg.get('warmup_epoch',1))
        self.learning_rate = cfg.get('lr', 1e-3)
        self.device = device
        self.continue_train = cfg.get('continue_train', False)
        self.pre_train = cfg.get('pre_train',False)
        self.cfg = cfg
        self.epochs_run = 0

        log_info(f'Device {self.device}',gpu_id=self.gpu_id)
        index_dir = os.path.join(DATA_PATHS[dataset], str(fold))
        data_path = os.path.join(DATA_PATHS[dataset], 'reviews_new.pickle')

        # self.load_data(data_path, index_dir, self.seq_mode, test_flg,load_direct=True)
        self.load_data_helper(data_helper)

        self.model_path = os.path.join(CKPT_PATH, f'{model_name}_{dataset}_{fold}.pt')
        self.prediction_path = prediction_path

        # PETER     :   [0,  1,  1, 1/0]
        # FMLP PETER:   [20, 0,  1, 1/0]
        if self.use_feature:
            self.src_len = self.hist_len + self.user_len+self.item_len + self.train_data.feature.size(1) # [u, i]
        else:
            self.src_len = self.hist_len + self.user_len+self.item_len
        log_info(f'Use Feature: {self.use_feature} | src_len: {self.src_len}',gpu_id=self.gpu_id)
        #               Feature, bos/eos, Text
        # PETER     :   [1,      15]
        # FMLP PETER:   [1,      15]
        # if self.use_feature:
        #     self.tgt_len = self.out_words + 1 + self.train_data.feature.size(1) # added <bos> or <eos>
        # else:
        #     self.tgt_len = self.out_words + 1  # added <bos> or <eos>
        
        self.tgt_len = self.out_words + 1
        if self.seq_mode >= HIST_REV_MODE:
            self.tgt_len += self.out_words * self.hist_len
        self.ntokens = self.corpus.ntokens
        self.nuser = self.corpus.nuser
        self.nitem = self.corpus.nitem
        
        self.history = TrainHistory(self.model_name)
        self.train_time = 0.

        self.build_model(model_name, cfg, self.device)
        self.text_criterion = nn.NLLLoss(ignore_index=self.word2idx[PAD_TOK])  # ignore the padding when computing loss
        self.nextit_criterion = nn.NLLLoss(ignore_index=self.word2idx[UNK_TOK])  # ignore the padding when computing loss
        self.rating_criterion = nn.MSELoss()

        if self.model is not None:
            if cfg.get('optimizer', 'SGD') == 'SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            elif cfg.get('optimizer', 'SGD') == 'AdamW':
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            
            if cfg.get('scheduler','steplr') == 'steplr':
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.25)
            elif cfg.get('scheduler','steplr') == 'warmup_cos':
                self.scheduler = get_cosine_schedule_with_warmup(
                        self.optimizer,         # 359303 / 64 / 4 * 5
                        num_warmup_steps=int(len(self.train_data) / self.batch_size / num_gpu * self.warmup_epoch),#135, # train data len / batch size / 4 * epochs
                        num_training_steps=int(len(self.train_data) / self.batch_size / num_gpu * self.epochs) )
            elif cfg.get('scheduler','steplr') == 'warmup_lin':
                self.scheduler = get_linear_schedule_with_warmup(
                        self.optimizer,         # 359303 / 64 / 4 * 5
                        num_warmup_steps=int(len(self.train_data) / self.batch_size / num_gpu * self.warmup_epoch),#135, # train data len / batch size / 4 * epochs
                        num_training_steps=int(len(self.train_data) / self.batch_size / num_gpu * self.epochs) )

        self.exp_metadata = {'Model': model_name, 'Dataset': dataset, 'Split_ID': int(fold), 
            'Num_GPU':int(num_gpu),'GPU_ID':int(self.gpu_id), 'HistLen': int(self.hist_len), 'TextLen':int(self.out_words),
            'Batch_size':int(self.batch_size),'Optimizer':cfg.get('optimizer', 'SGD'),'LR':self.learning_rate,'Seed':int(RNG_SEED)}

    def load_data_helper(self,data_helper):
        log_info(f'Loading data helper',gpu_id=self.gpu_id)
        # print(f'{now_time()} [GPU {self.gpu_id}] Loading data helper')
        self.corpus = data_helper['corpus']
        self.word2idx = data_helper['word2idx']
        self.idx2word = data_helper['idx2word']
        self.feature_set = data_helper['feature_set']
        self.train_data = data_helper['train_data']
        self.val_data = data_helper['val_data']
        self.test_data = data_helper['test_data']
        
        self.train_dataloader = data_helper['train_dataloader']
        self.val_dataloader = data_helper['val_dataloader']
        self.test_dataloader = data_helper['test_dataloader']

    def load_data(self, data_path, index_dir, seq_mode, test_flg=False,load_direct=False):
        log_info(f'Loading data {index_dir}',gpu_id=self.gpu_id)
        # print(f'{now_time()}[GPU {self.gpu_id}] Loading data {index_dir}')
        self.corpus = SEQUER_DataLoader(data_path, index_dir, self.vocab_size, seq_mode, test_flg,load_direct,self.hist_len)
        self.word2idx = self.corpus.word_dict.word2idx
        self.idx2word = self.corpus.word_dict.idx2word
        self.feature_set = self.corpus.feature_set

        # self.train_data = Batchify(self.corpus.train, self.word2idx, self.batch_size, seq_mode, shuffle=True)
        # self.val_data = Batchify(self.corpus.valid, self.word2idx, self.batch_size, seq_mode)
        # self.test_data = Batchify(self.corpus.test, self.word2idx, self.batch_size, seq_mode)

        self.train_data = AmazonDataset(self.corpus.train,self.word2idx,seq_mode)
        self.val_data = AmazonDataset(self.corpus.valid,self.word2idx,seq_mode)
        self.test_data = AmazonDataset(self.corpus.test,self.word2idx,seq_mode)

        self.train_dataloader = DataLoader(self.train_data,
                                            batch_size=self.batch_size, 
                                            pin_memory=True,
                                            shuffle=False,
                                            sampler=DistributedSampler(self.train_data))
        self.val_dataloader = DataLoader(self.val_data,batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_data,batch_size=self.batch_size)


    def build_model(self, model_name, cfg, device, multi_gpu=True):
        log_info(f'Building model {model_name.upper()} on device {device}',gpu_id=self.gpu_id)
        # print(now_time() + f'Building model {model_name.upper()}')
        self.model = None
        if 'peter' == model_name.lower():
            self.model = PETER(self.src_len, self.tgt_len, self.out_words, self.word2idx[PAD_TOK],
                               self.nuser, self.nitem, self.ntokens, cfg).to(self.gpu_id)
        elif 'sequer' == model_name.lower():
            self.model = SEQUER(self.src_len, self.tgt_len, self.out_words,self.word2idx[PAD_TOK],
                                self.nuser, self.nitem, self.ntokens, cfg).to(self.gpu_id)
        elif 'att2seq' == model_name.lower():
            self.model = Att2Seq(self.src_len, self.tgt_len, self.out_words,self.word2idx[PAD_TOK],
                                 self.nuser, self.nitem, self.ntokens, cfg).to(self.gpu_id)
        elif 'fmlpeter' == model_name.lower():
            self.model = FMLPETER(self.src_len, self.tgt_len, self.out_words, self.word2idx[PAD_TOK],
                                 self.nuser, self.nitem, self.ntokens, cfg,device,self.gpu_id).to(self.gpu_id)
        elif 'sawer' == model_name.lower():
            self.model = SAWER(self.src_len, self.tgt_len, self.out_words, self.word2idx[PAD_TOK],
                                 self.nuser, self.nitem, self.ntokens, cfg,device,self.gpu_id).to(self.gpu_id)
        elif 'pepler' == model_name.lower():
            self.model = PEPLER.from_pretrained('PEPLER/GPT2/', self.src_len, self.tgt_len, self.out_words, self.word2idx[PAD_TOK],
                                 self.nuser, self.nitem, self.ntokens, cfg).to(self.gpu_id)
            # self.model = ContinuousPromptLearning.from_pretrained('PEPLER/GPT2/', self.nuser, self.nitem)
            self.model.resize_token_embeddings(self.ntoken)  # three tokens added, update embedding table
        # self.model = self.model.to(self.gpu_id)
        
        if self.continue_train and os.path.exists(self.model_path):
            log_info(f"Loading checkpoint from {self.model_path}",gpu_id=self.gpu_id)
            self.load_checkpoint()

        if multi_gpu:
            self.model = DDP(self.model, device_ids=[self.gpu_id],output_device=self.gpu_id)

    def compute_loss(self, pred, labels):
        # c_loss, r_loss, t_loss, i_loss = 0, 0, 0, 0
        c_loss= torch.tensor(0, dtype=torch.int64).contiguous()
        r_loss= torch.tensor(0, dtype=torch.int64).contiguous()
        t_loss= torch.tensor(0, dtype=torch.int64).contiguous()
        i_loss= torch.tensor(0, dtype=torch.int64).contiguous()
        if 'context' in pred:
            # Pred [1920, 20004]  ;  Label [1920]  ;  15 * 128
            context_dis = pred['context'].unsqueeze(0).repeat((self.tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
            # log_info(f'[compute_loss] cnt loss: Input {context_dis.view(-1, self.ntokens).shape} Target {labels["seq"][1:-1].reshape((-1,)).shape}',gpu_id=self.gpu_id)
            c_loss = self.text_criterion(context_dis.view(-1, self.ntokens), labels['seq'][1:-1].reshape((-1,)))
        if 'rating' in pred:
            # Pred [128]  ;  Label [128]  ;  128
            # log_info(f'[compute_loss] rat loss: pred {pred["rating"].shape} label {labels["rating"].shape}',gpu_id=self.gpu_id)
            assert pred['rating'].shape == labels['rating'].shape
            r_loss = self.rating_criterion(pred['rating'], labels['rating'])
        if 'word' in pred:
            # Pred [2048, 20004]  ;  Label [2048];   16 * 128
            # log_info(f'[compute_loss] tgn loss: Input {pred["word"].view(-1, self.ntokens).shape} Target {labels["seq"][1:].reshape((-1,)).shape}',gpu_id=self.gpu_id)
            t_loss = self.text_criterion(pred['word'].view(-1, self.ntokens), labels['seq'][1:].reshape((-1,)))
        if pred.get('item', None) is not None:
            # Pred [2560, 7364]  ;  Label [2432] ;   20 * 128;  19 * 128
            # log_info(f'[compute_loss] item loss: Input {pred["item"].view(-1, self.nitem).shape} Target {labels["item_seq"].reshape((-1,)).shape}',gpu_id=self.gpu_id)
            i_loss = self.nextit_criterion(pred['item'].view(-1, self.nitem), labels['item_seq'].reshape((-1,)))
            

        # if 'item' in pred:
        #     log_info(f'item loss: Input {pred["item"].view(-1, self.nitem).shape} Target {labels["item_seq"][:, 1:].reshape((-1,)).shape}',gpu_id=self.gpu_id)
        #     i_loss = self.text_criterion(pred['item'].view(-1, self.nitem), labels['item_seq'][:, 1:].reshape((-1,)))
        # log_info(f'c_loss: {c_loss}, r_loss: {r_loss}, t_loss: {t_loss}, i_loss: {i_loss}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
        return c_loss, r_loss, t_loss, i_loss

    def train_per_epoch(self,ep):
        self.model.train()
        context_loss = 0.
        text_loss = 0.
        rating_loss = 0.
        item_loss = 0.
        total_sample = 0
        filter_last = (self.seq_mode == HIST_I_MODE)
        self.train_dataloader.sampler.set_epoch(ep)
        pbar = tqdm(total=len(self.train_dataloader), position=0, leave=True, ncols=150,bar_format=f'[Ep {ep:3d}: GPU {self.gpu_id}] '+'{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {elapsed} | {desc}')
        for step_i, batch in enumerate(self.train_dataloader):
            if self.pre_train=='GPT2':
                user, item, rating, seq, feature,mask = batch
            else:
                user, item, rating, seq, feature = batch
            batch_size = user.size(0)
            # log_info(f'[train_per_epoch] item size: {item.size()}',gpu_id=self.gpu_id)

            # print(f'user size: {user.size()}')
            user = user.to(self.gpu_id)  # (batch_size,)
            # Item : [128, 20]
            item = item.to(self.gpu_id)
            rating = rating.to(self.gpu_id)
            seq = seq.t().to(self.gpu_id)  # (tgt_len + 1, batch_size)
            if self.pre_train=='GPT2':
                mask = mask.to(self.device)
            
            # log_info(f'[train_per_epoch] Step:{step_i} rating label shape:{rating.shape}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
            labels = {'rating': rating, 'seq': seq, 'item_seq': item}
            feature = feature.t().to(self.gpu_id)  # (1, batch_size)
            if self.use_feature:
                text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
            else:
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
            if self.model_name.lower() == 'att2seq':
                text = text.permute(1,0)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.optimizer.zero_grad()

            # (TGTL, BSZ, NTOK) vs. (BSZ, NTOK) vs. (BSZ, HISTL+1,) vs (BSZ, HISTL, NITEM)
            # log_info(f'[train_per_epoch] user :{user.shape}, item :{item.shape}, text :{text.shape}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
            
            if self.pre_train=='GPT2':
                pred = self.model(user, item, text,mask=mask, filter_last=filter_last)
            else:
                pred = self.model(user, item, text, filter_last=filter_last)
            # log_info(f'[train_per_epoch] pred rating :{pred["rating"].shape}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
            c_loss, r_loss, t_loss, i_loss = self.compute_loss(pred, labels)
            loss = self.text_reg * t_loss + self.context_reg * c_loss + self.rating_reg * r_loss + self.item_reg * i_loss
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem.
            if self.clip_norm >0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()
            if self.cfg.get('scheduler','steplr') == 'warmup_cos' or self.cfg.get('scheduler','steplr') == 'warmup_lin':
                self.scheduler.step()

            # context_loss += batch_size * c_loss.item()
            # text_loss += batch_size * t_loss.item()
            # rating_loss += batch_size * r_loss.item()
            # log_info(f'c_loss type {type(c_loss)} t_loss type {type(t_loss)} r_loss type {type(r_loss)}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
            context_loss += batch_size * c_loss
            text_loss += batch_size * t_loss
            rating_loss += batch_size * r_loss
            item_loss += batch_size * i_loss
            total_sample += batch_size

            if step_i>0 and step_i % self.log_interval ==0:
                cur_c_loss = context_loss / total_sample
                cur_t_loss = text_loss / total_sample
                cur_r_loss = rating_loss / total_sample
                cur_i_loss = item_loss / total_sample

                if self.cfg.get('scheduler','steplr') == 'steplr':
                    lr = self.scheduler.get_last_lr()[0]
                else:
                    lr = self.optimizer.param_groups[0]["lr"]
                desc_str = f'context ppl {math.exp(cur_c_loss):4.4f} | text ppl {math.exp(cur_t_loss):4.4f} | rating loss {cur_r_loss:4.4f} | seq loss {cur_i_loss:4.4f} | LR {lr:1.8f}'

                pbar.set_description(desc_str)
                pbar.update(self.log_interval)
            
            
        pbar.close()
        cur_c_loss = context_loss / total_sample
        cur_t_loss = text_loss / total_sample
        cur_r_loss = rating_loss / total_sample
        cur_i_loss = item_loss / total_sample

        return cur_c_loss,cur_t_loss,cur_r_loss,cur_i_loss
    
    def evaluate_per_epoch(self,data):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        context_loss = 0.
        text_loss = 0.
        rating_loss = 0.
        item_loss = 0.
        total_sample = 0
        filter_last = (self.seq_mode == HIST_I_MODE)
        with torch.no_grad():
            for step_i, batch in enumerate(data):
                if self.pre_train=='GPT2':
                    user, item, rating, seq, feature, mask = batch
                else:
                    user, item, rating, seq, feature = batch

                batch_size = user.size(0)
                # log_info(f'[eval_per_epoch] user size: {user.size()}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
                user = user.to(self.gpu_id)  # (batch_size,)
                item = item.to(self.gpu_id)
                rating = rating.to(self.gpu_id)
                seq = seq.t().to(self.gpu_id)  # (tgt_len + 1, batch_size)
                labels = {'rating': rating, 'seq': seq, 'item_seq': item}
                # log_info(f'[eval_per_epoch] Step:{step_i} rating label shape:{rating.shape}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
                feature = feature.t().to(self.gpu_id)  # (1, batch_size)
                if self.pre_train=='GPT2':
                    mask = mask.to(self.gpu_id)
                if self.use_feature:
                    text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
                else:
                    text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
                if self.model_name.lower() == 'att2seq':
                    text = text.permute(1,0)
                # log_info(f'[eval_per_epoch] user :{user.shape}, item :{item.shape}, text :{text.shape}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
                
                if self.pre_train=='GPT2':
                    pred = self.model(user, item, text,mask=mask,filter_last=filter_last)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                else:
                    pred = self.model(user, item, text,filter_last=filter_last)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                # log_info(f'[eval_per_epoch] pred rating :{pred["rating"].shape}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
                c_loss, r_loss, t_loss, i_loss = self.compute_loss(pred, labels)

                context_loss += batch_size * c_loss.item()
                text_loss += batch_size * t_loss.item()
                rating_loss += batch_size * r_loss.item()
                item_loss += batch_size * i_loss.item()
                total_sample += batch_size
        return context_loss / total_sample, text_loss / total_sample, rating_loss / total_sample, item_loss / total_sample

    def generate_per_epoch(self,data):
        self.model.eval()
        self.model.seq_prediction = False  # Make sure it only predicts next token for the last token
        idss_predict = []
        context_predict = []
        rating_predict = []
        item_predict = []

        with torch.no_grad():
            for step_i, batch in enumerate(data):
                if self.pre_train=='GPT2':
                    user, item, rating, seq, feature,mask = batch
                else:
                    user, item, rating, seq, feature = batch
                user = user.to(self.gpu_id)  # (batch_size,)
                item = item.to(self.gpu_id)
                # for att2seq
                inputs = seq[:, 0].unsqueeze(0).to(self.gpu_id) # 1, 128
                # log_info(f'inputs {inputs.shape}',gpu_id=self.gpu_id)
                
                bos = seq[:, 0].unsqueeze(0).to(self.gpu_id)  # (1, batch_size)
                # log_info(f'bos {bos.shape}',gpu_id=self.gpu_id)
                
                feature = feature.t().to(self.gpu_id)  # (1, batch_size)
                if self.pre_train=='GPT2':
                    mask = mask.to(self.gpu_id)
                if self.use_feature:
                    text = torch.cat([feature, bos], 0)  # (src_len - 1, batch_size)
                else:
                    text = bos  # (src_len - 1, batch_size)
                # text_word = [self.idx2word[id] for id in text[:,0]]
                # log_info(f'text 0:{text_word}',gpu_id=self.gpu_id,level=LOG_DEBUG)
                # text_word = [self.idx2word[id] for id in text[:,5]]
                # log_info(f'text 5:{text_word}',gpu_id=self.gpu_id,level=LOG_DEBUG)
                # log_info(f'bos:{bos.shape},text:{text.size()},out_words:{self.out_words}',gpu_id=self.gpu_id,level=LOG_DEBUG)
                hidden = None
                hidden_c = None
                filter_last = (self.seq_mode == HIST_I_MODE)
                start_idx = text.size(0)
                inputs = inputs.permute(1,0)
                ids = inputs
                for idx in range(self.out_words):
                    # produce a word at each step
                    if self.model_name.lower() == 'att2seq':
                        # inputs = inputs.permute(1,0)
                        if idx == 0:
                            hidden = self.model.encoder(user, item)
                            hidden_c = torch.zeros_like(hidden)
                            log_info(f'user {user.shape} item {item.shape} input {inputs.shape}, hidden {hidden.shape}, hidden_c {hidden_c.shape}')
                            log_word_prob, hidden, hidden_c = self.model.decoder(inputs, hidden, hidden_c)  # (batch_size, 1, ntoken)
                            # 128,16,20004
                        else:
                            log_info(f'user {user.shape} item {item.shape} input {inputs.shape}, hidden {hidden.shape}, hidden_c {hidden_c.shape}')
                            log_word_prob, hidden, hidden_c = self.model.decoder(inputs, hidden, hidden_c)  # (batch_size, 1, ntoken)
                        word_prob = log_word_prob.squeeze().exp()  # (batch_size, ntoken)
                        log_info(f'word_prob :{word_prob.shape}, log_word {log_word_prob.shape}')
                        # log_info(f'word_prob :{word_prob}, log_word {log_word_prob}')
                        inputs = torch.argmax(word_prob, dim=-1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                        ids = torch.cat([ids, inputs], 1)  # (batch_size, len++)
                        log_info(f'step {step_i} idx {idx+1}/{self.out_words} : log_word_prob {log_word_prob.shape}, ids {ids.shape}',gpu_id=self.gpu_id)
                    else:
                        if idx == 0:
                            if self.pre_train=='GPT2':
                                pred = self.model.generate(user, item, text, mask=None, filter_last=filter_last)
                            else:    
                                pred = self.model.generate(user, item, text, filter_last=filter_last)
                                if 'rating' in pred:
                                    rating_predict.extend(pred['rating'])
                                if 'context' in pred:
                                    context_predict.extend(pred['context'])
                        else:
                            if self.pre_train=='GPT2':
                                pred = self.model(user, item, text,mask=mask)  # (batch_size, ntoken)
                            else:
                                pred = self.model(user, item, text)  # (batch_size, ntoken)
                        word_prob = pred['word'].exp()  # (batch_size, ntoken)
                        #                     1 , 128 , 20004 (text len, batchsize, ntoken)
                        # log_info(f'word_prob:{word_prob.shape}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
                        word_idx = torch.argmax(word_prob, dim=-1)  # (batch_size,), pick the one with the largest probability
                        # log_info(f'word_idx:{word_idx.shape}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
                        
                        text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
                        # text = torch.cat([text, word_idx[-1:,:]], 0)  # (len++, batch_size)
                        # new_word = self.idx2word[word_idx[-1]]
                        # text_word= [self.idx2word[id] for id in text[:,0]]
                        # log_info(f'new text 0:{new_word} \ntext 0:{text_word}',gpu_id=self.gpu_id)
                if self.model_name.lower() == 'att2seq':
                    ids = ids[:, 1:].tolist()  # remove bos
                    log_info(f'step {step_i} ids: {ids}')
                else:
                    ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
                idss_predict.extend(ids)

        # log_info(f'rating shape {data.dataset.rating.shape}',gpu_id=self.gpu_id,level=LOG_DEBUG)
        if len(data.dataset.rating.shape) == 2:
            data.dataset.data.rating = data.dataset.data.rating[:, -1]
        text_out = self.compute_metrics(data.dataset, rating_predict, idss_predict, context_predict)

        return text_out

    def compute_metrics(self, data, rating_predict, idss_predict, context_predict):
        results = self.exp_metadata.copy()
        results.update({m: 0 for m in METRICS})
        results['TestTime'] = now_time()
        
        # rating
        if rating_predict:
            predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
            log_info(f'predicted_rating {predicted_rating[:10]}',gpu_id=self.gpu_id,level=LOG_DEBUG)
            results['RMSE↓'] = root_mean_square_error(predicted_rating, self.corpus.max_rating, self.corpus.min_rating)
            log_info(f'RMSE↓ {results["RMSE↓"]:7.4f}',gpu_id=self.gpu_id)
            # print(now_time() + 'RMSE {:7.4f}'.format(results['RMSE']))
            results['MAE↓'] = mean_absolute_error(predicted_rating, self.corpus.max_rating, self.corpus.min_rating)
            log_info(f'MAE↓ {results["MAE↓"]:7.4f}',gpu_id=self.gpu_id)
            # print(now_time() + 'MAE {:7.4f}'.format(results['MAE']))
        # text
        tokens_test = [ids2tokens(ids[1:], self.word2idx, self.idx2word) for ids in data.seq.tolist()]
        tokens_predict = [ids2tokens(ids, self.word2idx, self.idx2word) for ids in idss_predict]
        log_info(f'idss_predict {len(idss_predict)}, tokens_predict {len(tokens_predict)}',gpu_id=self.gpu_id)
        results['BLEU_1↑'] = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
        log_info(f'BLEU_1↑ {results["BLEU_1↑"]:7.4f}',gpu_id=self.gpu_id)

        results['BLEU_4↑'] = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
        log_info(f'BLEU_4↑ {results["BLEU_4↑"]:7.4f}',gpu_id=self.gpu_id)
        
        results['USR↑'], results['USN↑'] = unique_sentence_percent(tokens_predict)
        log_info(f'USR↑ {results["USR↑"]:7.4f} | USN↑ {results["USN↑"]:7}',gpu_id=self.gpu_id)
        
        feature_batch = feature_detect(tokens_predict, self.feature_set)
        results['DIV↓'] = feature_diversity(feature_batch)  # time-consuming
        log_info(f'DIV↓ {results["DIV↓"]:7.4f}',gpu_id=self.gpu_id)
        
        results['FCR↑'] = feature_coverage_ratio(feature_batch, self.feature_set)
        log_info(f'FCR↑ {results["FCR↑"]:7.4f}',gpu_id=self.gpu_id)

        feature_test = [self.idx2word[i] for i in data.feature.squeeze(1).tolist()]  # ids to words
        results['FMR↑'] = feature_matching_ratio(feature_batch, feature_test)
        log_info(f'FMR↑ {results["FMR↑"]:7.4f}',gpu_id=self.gpu_id)

        
        text_test = [' '.join(tokens) for tokens in tokens_test]
        text_predict = [' '.join(tokens) for tokens in tokens_predict]
        ROUGE = rouge_score(text_test, text_predict)  # a dictionary
        for (k, v) in ROUGE.items():
            results[k] = v
            # print(now_time() + '{} {:7.4f}'.format(k, v))
            log_info(f'{k} {v:7.4f}',gpu_id=self.gpu_id)
        text_out = ''
        if self.gen_flg:
            save_results(results)
            if context_predict:
                tokens_context = [' '.join([self.idx2word[i] for i in ids]) for ids in context_predict]
                for (real, ctx, fake) in zip(text_test, tokens_context, text_predict):
                    text_out += f'True: {real}\nCntx: {ctx}\nPred: {fake}\n\n'
            else:
                for (real, fake) in zip(text_test, text_predict):
                    text_out += f'True: {real}\nPred: {fake}\n\n'
        log_info(f'Text Out: {text_out[:100]}',gpu_id=self.gpu_id,level=LOG_DEBUG)
        return text_out

    def train(self):
        best_val_loss = float('inf')
        endure_count = 0
        
        starttime = time.time()           
        for epoch in range(self.epochs_run, self.epochs + 1):
            curt = time.time()
            log_info(f'Epoch {epoch} Start',gpu_id=self.gpu_id)
            train_c_loss, train_t_loss, train_r_loss, train_i_loss = self.train_per_epoch(epoch)
            
            val_c_loss, val_t_loss, val_r_loss,val_i_loss = self.evaluate_per_epoch(self.val_dataloader)

            val_loss = math.exp(val_t_loss) * self.text_reg + math.exp(val_c_loss) * self.context_reg + val_r_loss * self.rating_reg + val_i_loss * self.item_reg
            train_loss = math.exp(train_t_loss) * self.text_reg + math.exp(train_c_loss) * self.context_reg + train_r_loss * self.rating_reg + train_i_loss * self.item_reg
            # if self.rating_reg == 0:
            #     val_loss = val_t_loss
            #     train_loss = train_t_loss
            # else:
            #     val_loss = val_t_loss + val_r_loss
            #     train_loss = train_t_loss + train_r_loss
        
            log_info(f'context ppl {math.exp(val_c_loss):4.4f} | text ppl {math.exp(val_t_loss):4.4f} | rating loss {val_r_loss:4.4f} | seq loss {val_i_loss:4.4f} | valid loss {val_loss:4.4f} on validation',gpu_id=self.gpu_id)

            # Save the model if the validation loss is the best we've seen so far.
            self.history.add_history(epoch=epoch, hist_dict={'train_context_loss':math.exp(train_c_loss),'train_text_loss':math.exp(train_t_loss), 
                        'train_rating_loss':train_r_loss.cpu().detach().numpy(),'train_loss':train_loss.cpu().detach().numpy(),
                        'val_context_loss':math.exp(val_c_loss),'val_text_loss':math.exp(val_t_loss),'val_rating_loss':val_r_loss,'val_loss':val_loss})
            # log_info(str(self.history),gpu_id=self.gpu_id)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.gpu_id == 0:
                    self.save_checkpoint(epoch)
                # with open(self.model_path, 'wb') as f:
                #     torch.save(self.model, f)
            else:
                endure_count += 1
                if endure_count == self.endure_times:
                    log_info(f'Endured {endure_count} / {self.endure_times} time(s)| Cannot endure it anymore | Exiting from early stop',gpu_id=self.gpu_id)
                    log_info(f'Epoch {epoch} End | Time Used: {time.time()-curt:.2f}\n',gpu_id=self.gpu_id)
                    log_info(f'Training Finished | Time Used: {time.time()-starttime:.2f}\n{"-"*80}',gpu_id=self.gpu_id)
                    break
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                if self.cfg.get('scheduler','steplr') == 'steplr':
                    self.scheduler.step()                                                               # self.scheduler.get_last_lr()[0]
                log_info(f'Endured {endure_count} / {self.endure_times} time(s) | Learning rate set to {self.optimizer.param_groups[0]["lr"]:2.8f}',gpu_id=self.gpu_id)
                
            log_info(f'Epoch {epoch} End | Time Used: {time.time()-curt:.2f}\n',gpu_id=self.gpu_id)
        self.exp_metadata['Total_Epoch'] = int(epoch)
        self.train_time += time.time()-starttime
        self.exp_metadata['Train_Time'] = f'{self.train_time:.2f}'
        self.exp_metadata['Epoch_Time'] = f'{self.train_time/epoch:.2f}'
        
        self.history.save_plot()
        return self.history
    
    def save_checkpoint(self,epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "LR":self.optimizer.param_groups[0]["lr"],
            "HISTORY": self.history,
            "Time":now_time(),
            "Train_Time":self.train_time
        }
        torch.save(snapshot, self.model_path)
        log_info(f"Epoch {epoch} | LR {self.optimizer.param_groups[0]['lr']} | Num History {len(self.history)} | Training snapshot saved at {self.model_path}",gpu_id=self.gpu_id)
        # with open(self.model_path, 'wb') as f:
        #     torch.save(self.model.module.state_dict(), f)

    def load_checkpoint(self):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(self.model_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.learning_rate = snapshot["LR"]
        self.history = snapshot['HISTORY']
        self.train_time = snapshot['Train_Time']

        log_info(f"Resuming training from snapshot at Epoch {self.epochs_run}, Num History {len(self.history)}",gpu_id=self.gpu_id)
        # with open(self.model_path, 'rb') as f:
        #     new_dict = torch.load(f)
        #     # self.model.module.load_state_dict(new_dict)
        #     self.model.load_state_dict(new_dict)
        #     self.model = self.model.to(self.gpu_id)

    def test(self):
        # Load the best saved model.
        self.build_model(self.model_name, self.cfg, self.device,multi_gpu=False)
        self.load_checkpoint()
        # Run on test data.
        # test_c_loss, test_t_loss, test_r_loss = self.evaluate(self.test_data)
        test_c_loss, test_t_loss, test_r_loss, test_i_loss = self.evaluate_per_epoch(self.test_dataloader)
        test_loss = math.exp(test_t_loss) * self.text_reg + math.exp(test_c_loss) * self.context_reg + test_r_loss * self.rating_reg + test_i_loss * self.item_reg
        
        log_info('=' * 89,gpu_id=self.gpu_id,gpu=False,time=False)
        log_info(f'context ppl {math.exp(test_c_loss):4.4f} | text ppl {math.exp(test_t_loss):4.4f} | rating loss {test_r_loss:4.4f} | seq loss {test_i_loss:4.4f} | loss {test_loss:4.4f} on test |End of training',gpu_id=self.gpu_id)

        log_info('Generating text',gpu_id=self.gpu_id)
        # text_o = self.generate(self.test_data)
        text_o = self.generate_per_epoch(self.test_dataloader)
        if self.gen_flg and self.gpu_id==0:
            with open(self.prediction_path, 'w', encoding='utf-8') as f:
                    f.write(text_o)
            log_info(f'Generated text saved to {self.prediction_path}',gpu_id=self.gpu_id)

    # def plot_history(self):        
    #     # log_info(f'{self.history}',gpu_id=self.gpu_id)
    #     # log_info(f"train loss: {self.history['train_loss']}", gpu_id=self.gpu_id)
    #     # log_info(f"val loss: {self.history['val_loss']}", gpu_id=self.gpu_id)
    #     hist_train_loss = self.history['train_loss']
    #     hist_val_loss = self.history['val_loss']

    #     dim = np.arange(1,len(hist_train_loss)+1,1)
    #     df_hist = pd.DataFrame(self.history,index=dim)
    #     df_hist.to_csv(os.path.join(RES_PATH,f'history-{self.model_name}.csv'))

    #     plt.plot(dim,hist_train_loss,label='train_loss')
    #     plt.plot(dim,hist_val_loss,label='val_loss')
    #     plt.title(f'{self.model_name} Loss')
    #     plt.ylabel('Loss')
    #     plt.xlabel('Epoch')
    #     plt.legend(loc='upper right')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(RES_PATH,f'history-{self.model_name}.jpg'))
    #     plt.close()