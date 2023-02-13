import math
import torch
import torch.nn as nn
import torch.nn.functional as func

from modules.base import MLP, PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, generate_peter_mask, \
    generate_square_subsequent_mask
from constants import *

class BaseModel(nn.Module):
    def __init__(self, src_len, tgt_len, word_len,pad_idx, nuser, nitem, ntoken, emsize):
        super(BaseModel, self).__init__()
        self.ui_len = 2
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.pad_idx = pad_idx
        self.word_len = word_len
        self.emsize = emsize

        self.user_embeddings = nn.Embedding(nuser, self.emsize)
        self.item_embeddings = nn.Embedding(nitem, self.emsize, padding_idx=self.pad_idx)
        self.word_embeddings = nn.Embedding(ntoken, self.emsize, padding_idx=self.pad_idx)
        self.hidden2token = nn.Linear(self.emsize, ntoken)
        self.recommender = MLP(self.emsize)

    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[0])  # (batch_size, ntoken)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, hidden):
        rating = self.recommender(hidden[0])  # (batch_size,)
        return rating

    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[self.src_len:])  # (tgt_len, batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def generate_token(self, hidden):
        word_prob = self.hidden2token(hidden[-1])  # (batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def predict_tasks(self, hidden):
        if self.rating_prediction:
            rating = self.predict_rating(hidden).T  # (batch_size, seq_len)
        else:
            rating = None
        if self.context_prediction:
            log_context_dis = self.predict_context(hidden)  # (batch_size, ntoken)
        else:
            log_context_dis = None
        if self.nextitem_prediction:
            log_next_item = self.predict_next(hidden)
        else:
            log_next_item = None
        if self.seq_prediction:
            log_word_prob = self.predict_seq(hidden)  # (tgt_len, batch_size, ntoken)
        else:
            log_word_prob = self.generate_token(hidden)  # (batch_size, ntoken)

        return rating, log_context_dis, log_next_item, log_word_prob

    def predict(self, log_context_dis, topk):
        word_prob = log_context_dis.exp()  # (batch_size, ntoken)
        if topk == 1:
            context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
        else:
            context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
        return context  # (batch_size, topk)

    def generate(self, user, item, text, **kwargs):
        pred = self(user, item, text, **kwargs)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
        return {'rating': pred['rating'].tolist(),
                'context': self.predict(pred['context'], topk=self.word_len).tolist(),
                'word': pred['word']}


class PETER(BaseModel):
    def __init__(self, src_len, tgt_len, word_len,pad_idx, nuser, nitem, ntoken, cfg):
        super().__init__(src_len, tgt_len, word_len,pad_idx, nuser, nitem, ntoken, cfg['emsize'])
        dropout = cfg.get('dropout', 0.5)
        nhead = cfg['nhead']
        nhid = cfg['nhid']
        nlayers = cfg['nlayers']
        peter_mask = cfg['peter_mask']
        self.seq_prediction = cfg['seq_prediction']
        self.context_prediction = cfg['context_prediction']
        self.rating_prediction = cfg['rating_prediction']
        self.nextitem_prediction = cfg['nextitem_prediction']
        self.use_feat = cfg.get('use_feature', True)

        self.pos_encoder = PositionalEncoding(self.emsize, dropout)  # emsize: word embedding size
        encoder_layers = TransformerEncoderLayer(self.emsize, nhead, nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above

        if peter_mask:
            self.attn_mask = generate_peter_mask(src_len, tgt_len,use_feat=self.use_feat,plot=True,filename=RES_PATH+f'/MASK_PETER{"+" if self.use_feat else ""}_PMask.png')
        else:
            self.attn_mask = generate_square_subsequent_mask(src_len + tgt_len,use_feat=self.use_feat,plot=True,filename=RES_PATH+f'/MASK_PETER{"+" if self.use_feat else ""}.png')

        self.init_weights()

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[1])  # (batch_size, ntoken)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis
        
    def forward(self, user, item, text, **kwargs):
        """
        :param user: (batch_size,), torch.int64
        :param item: (batch_size,), torch.int64
        :param text: (total_len - ui_len, batch_size), torch.int64
        :param seq_prediction: bool
        :param context_prediction: bool
        :param rating_prediction: bool
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken) if seq_prediction=True; the last token (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        :return attns: (nlayers, batch_size, total_len, total_len)
        """
        device = user.device
        batch_size = user.size(0)
        total_len = self.ui_len + text.size(0)  # deal with generation when total_len != src_len + tgt_len
        # see nn.MultiheadAttention for attn_mask and key_padding_mask
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)
        left = torch.zeros(batch_size, self.ui_len).bool().to(device)  # (batch_size, ui_len)
        right = text.t() == self.pad_idx  # replace pad_idx with True and others with False, (batch_size, total_len - ui_len)
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)

        u_src = self.user_embeddings(user.unsqueeze(0))  # (1, batch_size, emsize)
        i_src = self.item_embeddings(item.unsqueeze(0))  # (1, batch_size, emsize)
        w_src = self.word_embeddings(text)  # (total_len - ui_len, batch_size, emsize)
        src = torch.cat([u_src, i_src, w_src], 0)  # (total_len, batch_size, emsize)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden, attns = self.transformer_encoder(src, attn_mask, key_padding_mask)  # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)
        rating, log_context_dis, _, log_word_prob = self.predict_tasks(hidden)
        return {'word': log_word_prob, 'context': log_context_dis, 'rating': rating, 'attns': attns}
