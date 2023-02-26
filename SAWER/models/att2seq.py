import torch

from torch import nn
from .base import MLPEncoder, LSTMDecoder


class Att2Seq(nn.Module):
    def __init__(self, pad_idx, nuser, nitem, ntoken, cfg):  # nuser, nitem, ntoken, emsize, hidden_size, dropout, num_layers=2):  # src_len, tgt_len,
        super(Att2Seq, self).__init__()

        # NOTE: Despite having a num_layers parameter, it only serves as a multiplying factor to the hidden size
        self.encoder = MLPEncoder(nuser, nitem, cfg['emsize'], cfg['hidden_size'], cfg['num_layers'], pad_idx)
        self.decoder = LSTMDecoder(ntoken, cfg['hidden_size'], cfg['hidden_size'], cfg['num_layers'], cfg['dropout'],
                                   batch_first=False)

    def forward(self, user, item, seq):  # (batch_size,) vs. (seq_len, batch_size)
        h0 = self.encoder(user, item)  # (num_layers, batch_size, hidden_size)
        c0 = torch.zeros_like(h0)
        log_word_prob, _, _ = self.decoder(seq, h0, c0)

        return {'word': log_word_prob}

    def generate(self, user, item, text, **kwargs):
        hidden = None
        hidden_c = None
        inputs = text[:1]
        ids = inputs
        for idx in range(kwargs['out_words']):
            # produce a word at each step
            if idx == 0:
                hidden = self.encoder(user, item)
                hidden_c = torch.zeros_like(hidden)
                log_word_prob, hidden, hidden_c = self.decoder(inputs, hidden, hidden_c)  # (batch_size, 1, ntoken)
            else:
                # Possible error in Lileipisces implementation (text or ids?)
                log_word_prob, hidden, hidden_c = self.decoder(inputs, hidden, hidden_c)  # (batch_size, 1, ntoken)
            word_prob = log_word_prob.exp()  # (batch_size, ntoken)
            inputs = torch.argmax(word_prob, dim=-1)  # (batch_size, 1), pick the one with the largest probability
            ids = torch.cat([ids, inputs], 0)  # (batch_size, len++)
        ids = ids[1:].T.tolist()
        return {'context': [],
                'rating': [],
                'ids': ids}
