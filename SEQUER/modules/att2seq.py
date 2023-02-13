import torch

from torch import nn
from modules.base import MLPEncoder, LSTMDecoder
from modules.peter import BaseModel
from utils import log_info
from constants import *

class Att2Seq(BaseModel):
    def __init__(self, src_len, tgt_len, word_len, pad_idx, nuser, nitem, ntoken, cfg):  # nuser, nitem, ntoken, emsize, hidden_size, dropout, num_layers=2):  # src_len, tgt_len,
        super().__init__(src_len, tgt_len, word_len,pad_idx, nuser, nitem, ntoken, cfg['emsize'])

        self.encoder = MLPEncoder(nuser, nitem, cfg['emsize'], cfg['nhid'], cfg['nlayers'], pad_idx)
        self.decoder = LSTMDecoder(ntoken, cfg['nhid'], cfg['nhid'], cfg['nlayers'], cfg['dropout'])
        self.recommender = None
        self.user_embeddings = None
        self.item_embeddings = None
        self.word_embeddings = None
        self.hidden2token = None
                                                    #                   [16, 128] -> [128, 16]
    def forward(self, user, item, seq, **kwargs):  # (batch_size,) vs. (batch_size, seq_len)
        # log_info(f'user shape {user.shape}, item shape {item.shape}',level=LOG_DEBUG_DETAIL)
        # log_info(f'seq: {seq.shape}')
        h0 = self.encoder(user, item)  # (num_layers, batch_size, hidden_size)
        c0 = torch.zeros_like(h0)
        log_word_prob, _, _ = self.decoder(seq, h0, c0)
        #   128,16,20004
        return {'word': log_word_prob}

    def generate(self, user, item, text, **kwargs):
        return self(user, item, text)
