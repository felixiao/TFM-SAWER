import math
import torch
import torch.nn.functional as func
import os
from constants import CKPT_PATH,LOG_DEBUG_DETAIL,LOG_INFO,LOG_DEBUG,BASE_PATH,RES_PATH
import logging

from utils import log_info
from modules.base import PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, \
    generate_sawer_mask, MLPETER, MLP

from modules.peter import BaseModel
from modules.fmlp_modules import FMLPRecModel, FMLP_Args
from torch.nn.parallel import DistributedDataParallel as DDP

def load_fmlp(gpu_id,multi_gpu=True):
    args = FMLP_Args(gpu_id)
    # check the unique item count
    args.item_size = 7360 + 1
    model = FMLPRecModel(args=args)
    # trainer = FMLPRecTrainer(model, train_dataloader, eval_dataloader,test_dataloader, args)
    
    checkpoint_path = os.path.join(CKPT_PATH, args.load_model)
    log_info(f'LOAD FLMP MODEL {args.load_model}',gpu_id=gpu_id)
    # trainer.load(args.checkpoint_path)
    original_state_dict = model.state_dict()
    # print('original state',original_state_dict.keys())
    new_dict = torch.load(checkpoint_path)
    # print('new state',new_dict.keys())
    for key in new_dict:
        original_state_dict[key]=new_dict[key]
    model.load_state_dict(original_state_dict)
    # print(f'FMLP Model Load at {gpu_id}')
    model = model.to(gpu_id)
    if multi_gpu:
        model = DDP(model, device_ids=[gpu_id],output_device=gpu_id)
    return model

class SAWER(BaseModel):
    def __init__(self, src_len, tgt_len, word_len,pad_idx, nuser, nitem, ntoken, cfg,device,gpu_id,multi_gpu=True):
        super().__init__(src_len, tgt_len,word_len, pad_idx, nuser, nitem, ntoken, cfg['emsize'])
        dropout = cfg.get('dropout', 0.5)
        nhead = cfg['nhead']
        nhid = cfg['nhid']
        nlayers = cfg['nlayers']
        self.hist_len = cfg['hist_len']
        self.seq_prediction = cfg['seq_prediction']
        self.context_prediction = cfg['context_prediction']
        self.rating_prediction = cfg['rating_prediction']
        self.nextitem_prediction = cfg['nextitem_prediction']
        self.use_feat = cfg.get('use_feature', True)
        self.multi_gpu = multi_gpu
        self.gpu_id = gpu_id

        self.nuser = nuser
        self.nitem = nitem
        self.ntoken = ntoken
        log_info(f'[SAWER] nuser {nuser} | nitem {nitem} | ntoken {ntoken}',gpu_id=gpu_id)

        self.pos_encoder = PositionalEncoding(self.emsize, dropout)  # emsize: word embedding size
        encoder_layers = TransformerEncoderLayer(self.emsize, nhead, nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above
        self.recommender = MLP(self.emsize)

        self.fmlp_model = load_fmlp(gpu_id,multi_gpu)
        self.attn_mask = generate_sawer_mask(src_len, tgt_len,use_feat=self.use_feat,ver='Bert',plot= gpu_id==0,filename=os.path.join(RES_PATH,'MASK-SAWER_MaskBert.png'))
        self.init_weights()
        # self.user_embeddings = None
        self.item_embeddings = None

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[self.hist_len-1])  # (batch_size, n_token)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, hidden):
        rating = self.recommender(hidden[self.hist_len])  # (batch_size, seq_len)
        # log_info(f'[predict_rating] hidden shape: {hidden.shape} rating shape: {rating.shape}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
        #1 全部 hist item 和对应得分， 得出当前得分
        #2 item id emb + rating emb -> new item emb
        return rating

    def predict_next(self, hidden):
        log_info(f'PredNext: {hidden.shape}',gpu_id=self.gpu_id)
        if self.multi_gpu:
            log_next_item = func.log_softmax(torch.matmul(hidden[:self.hist_len-1], self.fmlp_model.module.item_embeddings.weight.T), dim=-1)
        else:
            log_next_item = func.log_softmax(torch.matmul(hidden[:self.hist_len-1], self.fmlp_model.item_embeddings.weight.T), dim=-1)
        # print(f'next item:{log_next_item.shape}, {log_next_item}')
        return log_next_item

    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[self.src_len:])  # (tgt_len, batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def forward(self, user, item, text, **kwargs):
        """
        :param user: (batch_size,), torch.int64
        :param item: (batch_size, seq_len), torch.int64
        :param text: (total_len - ui_len - hist_len, batch_size), torch.int64
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken) if seq_prediction=True; the last token (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        :return attns: (nlayers, batch_size, total_len, total_len)
        :return log_next_item: (batch_size, hist_len, nitem) if next_item_prediction=True; None otherwise
        :return
        """
        device = user.device
        batch_size = user.size(0)
        # log_info(f'batch_size:{batch_size}',gpu_id= self.gpu_id, level=LOG_DEBUG_DETAIL)
        #  38     =    1+1+15         22-1    
        #          feat +seq +eos       hist + item + feat -feat
        total_len =  text.size(0) + self.src_len -1 # deal with generation when total_len != src_len + tgt_len

        # log_info(f'src_len:{self.src_len},total_len:{total_len}',gpu_id= self.gpu_id)
        
        # see nn.MultiheadAttention for attn_mask and key_padding_mask
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)
        #                                   22-1
        left = torch.zeros(batch_size, self.src_len -1).bool().to(device)  # (batch_size, ui_len)
        
        #               17
        right = text.t() == self.pad_idx  # replace pad_idx with True and others with False, (batch_size, total_len - ui_len)
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)
        
        # log_info(f'left:{left.shape[1]}, right:{right.shape[1]},src_len:{self.src_len}, total_len:{total_len}, attn_mask: {attn_mask.shape} key_padding_mask: {key_padding_mask.shape}\n',
                # gpu_id=self.gpu_id, level=LOG_INFO)

        u_src = self.user_embeddings(user.unsqueeze(0))  # (1, batch_size, emsize)
        
        h_src = self.fmlp_model(item).permute(1,0,2)[:-1,:,:] # (batch_size, max_seq_len, emsize)       ---> (max_seq_len,batch_size,emsize) 
        if self.multi_gpu:
            i_src = self.fmlp_model.module.item_embeddings(item[:,-1:]).permute(1,0,2) # (batch_size, 1, emsize) ---> (1,          batch_size,emsize)
        else:
            i_src = self.fmlp_model.item_embeddings(item[:,-1:]).permute(1,0,2) # (batch_size, 1, emsize) ---> (1,          batch_size,emsize)
        
        w_src = self.word_embeddings(text)  # (total_len - ui_len - hist_len, batch_size, emsize)

        log_info(f'Hist: {h_src.shape}, Item:{i_src.shape}, User:{u_src.shape}, Word:{w_src.shape}',gpu_id=self.gpu_id)
        
        src = torch.cat([h_src,i_src, u_src, w_src], 0)  # (total_len, batch_size, emsize)
        # src = torch.cat([i_src, w_src], 0)  # (total_len, batch_size, emsize)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        # every time we run the transformer encoder, we obtain a different output for the same input
        
        hidden, attns = self.transformer_encoder(src, attn_mask, key_padding_mask)  # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)
        rating, log_context_dis, log_next_item, log_word_prob = self.predict_tasks(hidden)
        
        # log_info(f'[forward] rating shape:{rating.shape}, rating:{rating}',gpu_id=self.gpu_id)
        if 'filter_last' in kwargs.keys() and kwargs['filter_last']:
            rating = rating[-1]
        # log_info(f'[forward] rating shape:{rating.shape}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
        return {'word': log_word_prob, 'context': log_context_dis, 'rating': rating, 'item': log_next_item,
                'attns': attns}
