# @Time   : 2022/2/13
# @Author : Hui Yu
# @Email  : ishyu@outlook.com

import os
import copy
import math
import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import datetime
import numpy as np
import random
import pandas as pd

def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4] + ']: '

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

class FMLP_Args():

    def __init__(self,gpu_id):
        self.data_dir='./Data/Amazon/MoviesAndTV_New/'
        self.output_dir='FMLPETER/output/'
        self.data_name = 'reviews_new'
        self.do_eval= True
        self.load_model='FMLPRec-Movie_and_TV_index-Jan-04-2023_04-12-03_max20item.pt'
        
        self.model_name = 'FMLPRec'
        self.hidden_size = 512 # 64
        self.num_hidden_layers =4
        self.num_attention_heads=2
        self.hidden_act = 'gelu' # relu
        self.attention_probs_dropout_prob=0.5
        self.hidden_dropout_prob=0.5
        self.initializer_range=0.02
        self.max_seq_length=20
        self.no_filters=False

        self.lr = 0.001
        self.batch_size=512 #256
        self.epochs = 100 #200
        self.no_cuda= False
        self.log_freq=1
        self.full_sort=False
        self.patience=5

        self.seed=42
        self.weight_decay =0.0
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.99
        self.gpu_id = gpu_id
        self.variance = 5
        self.cuda_condition = torch.cuda.is_available() and not self.no_cuda
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        # save model args
        if self.no_filters:
            self.model_name = "SASRec"
        
        args_str = f'{self.model_name}-{self.data_name}-{now_time()[1:-3]}.txt'
        self.log_file = os.path.join(self.output_dir, args_str)
    
    def __str__(self):
        arg_str = ''
        for arg in vars(self):
            arg_str += f'{arg:30} {getattr(self, arg)}\n'
        return (
            f'{"FMLP ARGUMENTS":-^80}'
            f"{arg_str}"
            f'{"FMLP ARGUMENTS":-^80}'
        )
    __repr__ = __str__

def get_fmlp_model():
    args = FMLP_Args()
    print(args)

    set_seed(args.seed)
    check_path(args.output_dir)

    args.data_file = args.data_dir + args.data_name + '.pickle'
    peter_data = pd.DataFrame.from_records(pd.read_pickle(args.data_file))
    user_seq = peter_data['hist_item_index']
    sample_seq = peter_data['neg_samples_index']
    neg_item = peter_data['neg_item_index']
    num_users = len(peter_data)
    max_item = len(peter_data['item'].unique())

    # user_seq, max_item, num_users, sample_seq ,neg_item = get_user_seqs_and_sample_from_pickle(args.data_file)
    # seq_dic = {'user_seq':user_seq, 'num_users':num_users, 'sample_seq':sample_seq,'neg_item':neg_item}

    args.item_size = max_item + 1

    # train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)
    train_dataset = FMLPRecDataset(args, user_seq,neg_item, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = FMLPRecDataset(args, user_seq,neg_item,test_neg_items=sample_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = FMLPRecDataset(args, user_seq,neg_item,test_neg_items=sample_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = FMLPRecModel(args=args)
    trainer = FMLPRecTrainer(model, train_dataloader, eval_dataloader,test_dataloader, args)

    args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
    # trainer.load(args.checkpoint_path)
    original_state_dict = model.state_dict()
    # print('original state',original_state_dict.keys())
    new_dict = torch.load(args.checkpoint_path)
    # print('new state',new_dict.keys())
    for key in new_dict:
        original_state_dict[key]=new_dict[key]
    model.load_state_dict(original_state_dict)

    print(f"Load model from {args.checkpoint_path} for test!")
    # print(model)
    return model

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set([actual[i]])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len([actual[user_id]]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set([actual[user_id]])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FilterLayer(nn.Module):
    def __init__(self, args):
        super(FilterLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
        # print('FilterLayer input shape',input_tensor.shape)
        batch, seq_len, hidden = input_tensor.shape
        # print(f'batch:{batch}, seq_len:{seq_len}, hidden:{hidden}')
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        # print('weight shape=',weight.shape)
        # print('x shape = ', x.shape)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(4 * args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.no_filters = args.no_filters
        if self.no_filters:
            self.attention = SelfAttention(args)
        else:
            self.filterlayer = FilterLayer(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        if self.no_filters:
            hidden_states = self.attention(hidden_states, attention_mask)
        else:
            hidden_states = self.filterlayer(hidden_states)

        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class FMLPRecModel(nn.Module):
    def __init__(self, args):
        super(FMLPRecModel, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)

        self.apply(self.init_weights)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        # print(f'sequence device : {sequence.device}')
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # same as SASRec
    def forward(self, input_ids):
        device = input_ids.device
        # print(f'FMLP Forward Device: {device}')
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda(device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        sequence_output = item_encoded_layers[-1]

        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        if full_sort:
            self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        if full_sort:
            self.args.train_matrix = self.args.test_rating_matrix
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        print(original_state_dict.keys())
        new_dict = torch.load(file_name)
        print(new_dict.keys())
        for key in new_dict:
            original_state_dict[key]=new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        # [batch hidden_size]
        #pos = pos_emb.view(-1, pos_emb.size(2))
        #neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out[:, -1, :] # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg_emb * seq_emb, -1)
        #istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.mean(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        )# / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class FMLPRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FMLPRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm(enumerate(dataloader),
                                  desc="Rec EP_%s:%d" % (str_code, epoch),
                                  ncols=120,
                                  total=len(dataloader))
                                #   bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, answer, neg_answer = batch
                # Binary cross_entropy
                sequence_output = self.model(input_ids)
                
                loss = self.cross_entropy(sequence_output, answer, neg_answer)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, answers, _, neg_answer = batch
                    recommend_output = self.model(input_ids)
                    recommend_output = recommend_output[:, -1, :]# 推荐的结果
                    
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, answers, _, sample_negs = batch

                    # self.model.item_embeddings.register_forward_hook(get_activation('item_embeddings'))
                    
                    recommend_output = self.model(input_ids)
                    # print(activation['item_embeddings'])
                    # print('item embedding', activation['item_embeddings'].shape)
                    # print('item embedding self',self.model.item_embeddings(input_ids)[0])
                    # print('item embedding 0', activation['item_embeddings'][0])

                    # print('output', recommend_output.shape)
                    # print('output 0', recommend_output)
                    test_neg_items = torch.cat((answers.unsqueeze(-1), sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)

class FMLPRecDataset(Dataset):
    def __init__(self, args, user_seq, neg_item, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.neg_item = neg_item
        self.max_len = args.max_seq_length
        self.test_neg_items = test_neg_items
        self.data_type = data_type

        if data_type=='train':
            for seq in user_seq:
                input_ids = seq[-(self.max_len + 2):-2]  # keeping same as train set
                self.user_seq.append(input_ids)
                # for i in range(1,len(input_ids)+1):
                #     # self.user_seq.append(input_ids[:i + 1])
                #     self.user_seq.append(input_ids[:i])
        elif data_type=='valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:-1])
        else:
            self.user_seq = user_seq


    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        items = self.user_seq[index]
        input_ids = items[:-1]
        answer = items[-1]

        # seq_set = set(items)
        # neg_answer = neg_sample(seq_set, self.args.item_size)
        neg_answer = self.neg_item[index]

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]
        assert len(input_ids) == self.max_len
        # Associated Attribute Prediction
        # Masked Attribute Prediction

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                #torch.tensor(attribute, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                #torch.tensor(attribute, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
            )

        return cur_tensors

def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item