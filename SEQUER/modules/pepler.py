import math
import torch
import torch.nn.functional as func
import os
from constants import CKPT_PATH,LOG_DEBUG_DETAIL,LOG_INFO,LOG_DEBUG,BASE_PATH,RES_PATH

from transformers import GPT2LMHeadModel
from utils import log_info
from modules.base import PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, generate_sequer_mask,generate_fmlpeter_mask, MLPETER,MLP
from modules.peter import BaseModel


class PEPLER(BaseModel, GPT2LMHeadModel):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, src_len, tgt_len, word_len,pad_idx, nuser, nitem, ntoken, cfg,multi_gpu=True, freezeLM=True, **kwargs):
        # super(BaseModel, self).__init__(src_len, tgt_len,word_len, pad_idx, nuser, nitem, ntoken, cfg['emsize'])
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        self.ui_len = 2
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.pad_idx = pad_idx
        self.word_len = word_len
        
        dropout = cfg.get('dropout', 0.5)
        self.hist_len = cfg['hist_len']

        self.seq_prediction = cfg['seq_prediction']
        self.context_prediction = cfg['context_prediction']
        self.rating_prediction = cfg['rating_prediction']
        self.nextitem_prediction = cfg['nextitem_prediction']

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(nuser, nitem)
        model.resize_token_embeddings(self.ntoken)
        return model

    def init_prompt(self, nuser, nitem):
        emsize = self.transformer.wte.weight.size(1)  # 768
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

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

        # embeddings
        u_src = self.user_embeddings(user.unsqueeze(0)).permute(1,0,2)  # (batch_size, emsize)
        i_src = self.item_embeddings(item.unsqueeze(0)).permute(1,0,2)  # (batch_size, emsize)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        src = torch.cat([u_src, i_src, w_src], 0)  # (batch_size, total_len, emsize)

        if mask is None:
            # auto-regressive generation
            log_word_prob = super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            log_word_prob = super(GPT2LMHeadModel).forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)

        # log_info(f'[forward] rating shape:{rating.shape}',gpu_id=self.gpu_id,level=LOG_DEBUG_DETAIL)
        return {'word': log_word_prob}
