import math
import torch
import torch.nn.functional as func

from constants import HIST_LEN
from modules.base import PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, generate_sequer_mask
from modules.peter import BaseModel


class SEQUER(BaseModel):
    def __init__(self, src_len, tgt_len,word_len, pad_idx, nuser, nitem, ntoken, cfg):
        super().__init__(src_len, tgt_len,word_len, pad_idx, nuser, nitem, ntoken, cfg['emsize'])
        dropout = cfg.get('dropout', 0.5)
        nhead = cfg['nhead']
        nhid = cfg['nhid']
        nlayers = cfg['nlayers']
        self.seq_prediction = cfg['seq_prediction']
        self.context_prediction = cfg['context_prediction']
        self.rating_prediction = cfg['rating_prediction']
        self.nextitem_prediction = cfg['nextitem_prediction']

        self.pos_encoder = PositionalEncoding(self.emsize, dropout)  # emsize: word embedding size
        encoder_layers = TransformerEncoderLayer(self.emsize, nhead, nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above

        self.attn_mask = generate_sequer_mask(src_len, tgt_len)
        self.init_weights()

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[0])  # (batch_size, n_token)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, hidden):
        # rating = self.recommender(hidden[1:HIST_LEN+2])  # (batch_size, seq_len)
        rating = self.recommender(hidden[HIST_LEN+1:HIST_LEN+2])  # (batch_size, seq_len)
        return rating

    def predict_next(self, hidden):
        print(f'predict_next: FMLPETER')
        log_next_item = func.log_softmax(torch.matmul(hidden[1:HIST_LEN+1], self.item_embeddings.weight.T), dim=-1)
        return log_next_item

    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[self.src_len+HIST_LEN:])  # (tgt_len, batch_size, ntoken)
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
        total_len = self.ui_len + text.size(0) + HIST_LEN  # deal with generation when total_len != src_len + tgt_len
        # see nn.MultiheadAttention for attn_mask and key_padding_mask
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)
        left = torch.zeros(batch_size, 1).bool().to(device)  # (batch_size, ui_len)
        middle = item == self.pad_idx
        right = text.t() == self.pad_idx  # replace pad_idx with True and others with False, (batch_size, total_len - ui_len)
        key_padding_mask = torch.cat([left, middle, right], 1)  # (batch_size, total_len)
        # plot_mask(key_padding_mask[:10, :].cpu())
        # plot_mask(attn_mask.cpu())
        # plot_mask(torch.bitwise_or(attn_mask, key_padding_mask[0, :].unsqueeze(0).repeat(total_len, 1)).cpu())

        u_src = self.user_embeddings(user.unsqueeze(0))  # (1, batch_size, emsize)
        i_src = self.item_embeddings(item).permute(1, 0, 2)  # (seq_len, batch_size, emsize)
        w_src = self.word_embeddings(text)  # (total_len - ui_len - hist_len, batch_size, emsize)
        src = torch.cat([u_src, i_src, w_src], 0)  # (total_len, batch_size, emsize)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        # every time we run the transformer encoder, we obtain a different output for the same input
        hidden, attns = self.transformer_encoder(src, attn_mask, key_padding_mask)  # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)
        rating, log_context_dis, log_next_item, log_word_prob = self.predict_tasks(hidden)
        if kwargs['filter_last']:
            rating = rating[-1]
        return {'word': log_word_prob, 'context': log_context_dis, 'rating': rating, 'item': log_next_item,
                'attns': attns}
