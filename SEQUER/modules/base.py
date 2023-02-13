import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import Tuple, Optional
from torch import Tensor

from constants import *
from utils import plot_mask,log_info


class MLPEncoder(nn.Module):
    def __init__(self, nuser, nitem, emsize, hidden_size, nlayers, pad_idx):
        super(MLPEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize, padding_idx=pad_idx)
        self.encoder = nn.Linear(emsize * 2, hidden_size * nlayers)
        self.tanh = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()

    def forward(self, user, item):  # (batch_size,)
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)
        
        ui_concat = torch.cat([u_src, i_src], 1)  # (batch_size, emsize * 2)
        # log_info(f'u_src:{u_src.shape}, i_src:{i_src.shape}, ui_concat:{ui_concat.shape}',level=LOG_DEBUG_DETAIL)

        hidden = self.tanh(self.encoder(ui_concat))  # (batch_size, hidden_size * nlayers)
        # log_info(f'hidden:{hidden.shape}',level=LOG_DEBUG_DETAIL)
        state = hidden.reshape((-1, self.nlayers, self.hidden_size)).permute(1, 0, 2).contiguous()  # (num_layers, batch_size, hidden_size)
        # log_info(f'state:{state.shape}',level=LOG_DEBUG_DETAIL)
        return state


class LSTMDecoder(nn.Module):
    def __init__(self, ntoken, emsize, hidden_size, num_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.lstm = nn.LSTM(emsize, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.08
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, seq, ht, ct):  # seq: (batch_size, seq_len), ht & ct: (nlayers, batch_size, hidden_size)
        # log_info(f'seq: {seq.shape}') # [16, 128]
        # seq = seq.permute(1,0)
        # log_info(f'seq: {seq.shape}') # [16, 128]
        seq_emb = self.word_embeddings(seq)  # (batch_size, seq_len, emsize) [16, 128] -> [16, 128, 2048]
        # log_info(f'seq_emb: {seq_emb.shape}, ht: {ht.shape}, ct: {ct.shape}') 
        #   [16, 128, 2048]   ->    [128, 16, 2048]       [2, 128, 2048] -> 
        output, (ht, ct) = self.lstm(seq_emb, (ht, ct))  # (batch_size, seq_len, hidden_size) vs. (nlayers, batch_size, hidden_size)
        # log_info(f'output: {output.shape}')# 
        decoded = self.linear(output)  # (batch_size, seq_len, ntoken)
        # log_info(f'decoded: {decoded.shape}')
        return func.log_softmax(decoded, dim=-1), ht, ct


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = func.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        attns = []

        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)
        attns = torch.stack(attns)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return func.relu
    elif activation == "gelu":
        return func.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # d_model: word embedding size
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,) -> (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
        '''
        probably to prevent from rounding error
        e^(idx * (-log 10000 / d_model)) -> (e^(log 10000))^(- idx / d_model) -> 10000^(- idx / d_model) -> 1/(10000^(idx / d_model))
        since idx is an even number, it is equal to that in the formula
        '''
        pe[:, 0::2] = torch.sin(position * div_term)  # even number index, (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # odd number index
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, d_model) -> (1, max_len, d_model) -> (max_len, 1, d_model)
        # self.register_buffer('pe', pe)  # will not be updated by back-propagation, can be called via its name
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))# will not be updated by back-propagation, can be called via its name

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MLPETER(nn.Module):
    def __init__(self, emsize=512,seq_len=1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(emsize*seq_len, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize); (20,batch_size,emsize)
        # print(hidden.shape)
        hidden = hidden.transpose(0,1)
        # print(hidden.shape)
        hidden = self.flatten(hidden)
        # print(hidden.shape)
        mlp_vector = self.sigmoid(self.linear1(hidden))  # (batch_size, emsize)
        rating = torch.squeeze(self.linear2(mlp_vector))  # (batch_size,)
        return rating

class MLP(nn.Module):
    def __init__(self, emsize=512):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize)
        mlp_vector = self.sigmoid(self.linear1(hidden))  # (batch_size, seq_len, emsize)
        rating = torch.squeeze(self.linear2(mlp_vector))  # (batch_size, seq_len)
        return rating


def generate_square_subsequent_mask(total_len,plot=False,filename=RES_PATH+'/MASK_SQUARE.png'):
    mask = torch.tril(torch.ones(total_len, total_len))  # (total_len, total_len), lower triangle -> 1.; others 0.
    mask = mask == 0  # lower -> False; others True
    log_info(f'MASK_SQUARE Mask: Len={total_len}',level=LOG_DEBUG)
    if plot:
        plot_mask(mask,filename)
    return mask


def generate_peter_mask(src_len, tgt_len,use_feat,plot=False,filename=RES_PATH+'/MASK_PETER.png'):
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    mask[0, 1] = False  # allow to attend for user and item
    log_info(f'MASK_PETER Mask: Len={total_len}',level=LOG_DEBUG)
    if plot:
        label = f'| 1 User | 2 : Item | {"" if not use_feat else "3 : Feature (Opt) | "}{mask.shape[1]-tgt_len+1}-{mask.shape[1]}: Word Seq (w/ bos/eos) |'
        plot_mask(mask,filename,label=label)
    return mask

def generate_sawer_mask_new(src_len, tgt_len,use_feat,plot=False,filename=RES_PATH+'/MASK_SWAER_MASK2.png'):
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    mask[HIST_LEN-1, HIST_LEN] =False
    mask[HIST_LEN+1:,:HIST_LEN-1] = True
    
    log_info(f'MASK_SAWER_1 Mask: Len={total_len}',level=LOG_DEBUG)
    if plot:
        # label = f'| 1-{src_len-2 if use_feat else src_len-1}: History Item | {src_len-1 if use_feat else src_len}: Current Item | {"" if not use_feat else str(src_len)+" : Feature (Opt) | "}{mask.shape[1]-tgt_len+1}-{mask.shape[1]}: Word Seq (w/ bos/eos) |'
        label = f'| 1-{src_len-2 if use_feat else src_len-1}: History Item | {src_len-1 if use_feat else src_len}:User | {"" if not use_feat else str(src_len)+" : Feature (Opt) | "}{mask.shape[1]-tgt_len+1}-{mask.shape[1]}: Word Seq (w/ bos/eos) |'
        plot_mask(mask,filename,label=label)
        
    return mask

def generate_sawer_mask(src_len, tgt_len,use_feat,ver='1',plot=False,filename=RES_PATH+'/MASK_SWAER_MASK.png'):
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    if ver=='1':
        mask[HIST_LEN-1, HIST_LEN] =False
    elif ver =='2':
        mask[HIST_LEN-1, HIST_LEN] =False
        mask[HIST_LEN+1:,:HIST_LEN-1] = True
    elif ver =='Bert':
        if use_feat:
            mask[:src_len-1, :src_len-1] = False  # allow to attend for user and item
        else:
            mask[:src_len, :src_len] = False  # allow to attend for user and item
    elif ver =='Bert2':
        mask[HIST_LEN+1:,:HIST_LEN-1] = True
        if use_feat:
            mask[:src_len-1, :src_len-1] = False  # allow to attend for user and item
        else:
            mask[:src_len, :src_len] = False  # allow to attend for user and item
    elif ver =='3':
        mask[:, HIST_LEN] =False
    elif ver =='4':
        mask[:, HIST_LEN] =False
        mask[HIST_LEN+1:,:HIST_LEN-1] = True
    elif ver == '5':
        if use_feat:
            mask[:src_len-1, :src_len-1] = False  # allow to attend for user and item
        else:
            mask[:src_len, :src_len] = False  # allow to attend for user and item
    
    log_info(f'MASK_SAWER_1 Mask {ver}: Len={total_len}',level=LOG_DEBUG)
    if plot:
        # label = f'| 1-{src_len-2 if use_feat else src_len-1}: History Item | {src_len-1 if use_feat else src_len}: Current Item | {"" if not use_feat else str(src_len)+" : Feature (Opt) | "}{mask.shape[1]-tgt_len+1}-{mask.shape[1]}: Word Seq (w/ bos/eos) |'
        if ver == '5':
            label = f'| 1 User | 2-{src_len-1 if use_feat else src_len}: History Item | {"" if not use_feat else str(src_len)+" : Feature (Opt) | "}{mask.shape[1]-tgt_len+1}-{mask.shape[1]}: Word Seq (w/ bos/eos) |'
            plot_mask(mask,filename,label=label,ver='UI')
        else:
            label = f'| 1-{src_len-2 if use_feat else src_len-1}: History Item | {src_len-1 if use_feat else src_len}:User | {"" if not use_feat else str(src_len)+" : Feature (Opt) | "}{mask.shape[1]-tgt_len+1}-{mask.shape[1]}: Word Seq (w/ bos/eos) |'
            plot_mask(mask,filename,label=label,ver='IU')
        
    return mask

def generate_fmlpeter_mask_new(src_len, tgt_len,use_feat,plot=False,filename=RES_PATH+'/MASK_FMLP_PETER_new.png'):
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    # src len = 22; HISTLEN 20+1
    if use_feat:
        mask[0, :src_len-1] = False  # allow to attend for user and item
        # mask[1:src_len-1,0] = True
    else:
        mask[0, :src_len] = False  # allow to attend for user and item
        # mask[1:src_len-1,0] = True  # allow to attend for user and item
    log_info(f'MASK_FMLP_PETER Mask: Len={total_len}',level=LOG_DEBUG)
    if plot:
        # label = f'| 1-{src_len-2 if use_feat else src_len-1}: History Item | {src_len-1 if use_feat else src_len}: Current Item | {"" if not use_feat else str(src_len)+" : Feature (Opt) | "}{mask.shape[1]-tgt_len+1}-{mask.shape[1]}: Word Seq (w/ bos/eos) |'
        label = f'| 1: User | 2-{src_len-2 if use_feat else src_len-1}: History Item | {"" if not use_feat else str(src_len)+" : Feature (Opt) | "}{mask.shape[1]-tgt_len+1}-{mask.shape[1]}: Word Seq (w/ bos/eos) |'
        plot_mask(mask,filename,label=label)
        
    return mask

def generate_fmlpeter_mask(src_len, tgt_len,use_feat,plot=False,filename=RES_PATH+'/MASK_FMLP_PETER.png'):
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    # src len = 22; HISTLEN 20+1
    if use_feat:
        mask[:src_len-1, :src_len-1] = False  # allow to attend for user and item
    else:
        mask[:src_len, :src_len] = False  # allow to attend for user and item
    log_info(f'MASK_FMLP_PETER Mask: Len={total_len}',level=LOG_DEBUG)
    if plot:
        # label = f'| 1-{src_len-2 if use_feat else src_len-1}: History Item | {src_len-1 if use_feat else src_len}: Current Item | {"" if not use_feat else str(src_len)+" : Feature (Opt) | "}{mask.shape[1]-tgt_len+1}-{mask.shape[1]}: Word Seq (w/ bos/eos) |'
        label = f'| 1: User | 2-{src_len-2 if use_feat else src_len-1}: History Item | {"" if not use_feat else str(src_len)+" : Feature (Opt) | "}{mask.shape[1]-tgt_len+1}-{mask.shape[1]}: Word Seq (w/ bos/eos) |'
        plot_mask(mask,filename,label=label)
        
    return mask

def generate_sequer_mask(src_len, tgt_len,use_feat=False, plot=False,filename=RES_PATH+'/MASK_SEQUER.png'):
    total_len = src_len + HIST_LEN + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    mask[0, :HIST_LEN + 2] = False  # allow user to attend the whole item sequence
    mask[1:HIST_LEN + 1, 0] = True
    log_info(f'MASK_SEQUER Mask: Len={total_len}',level=LOG_DEBUG)
    if plot:
        plot_mask(mask,filename)
    return mask