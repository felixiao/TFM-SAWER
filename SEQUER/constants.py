import os

BASE_PATH = './SEQUER/'
DATA_PATHS = {
    'amazon-movies': './Data/Amazon/MoviesAndTV_New',
    'yelp': os.path.join(BASE_PATH, 'data/Yelp'),
    'tripadvisor': os.path.join(BASE_PATH, 'data/TripAdvisor'),
    'amazon-toys': './Data/Amazon/ToysAndGames',
    'amazon-sports':'./Data/Amazon/SportsAndOutdoors'
}
RES_PATH = os.path.join(BASE_PATH, 'results')
CKPT_PATH = os.path.join(BASE_PATH, 'checkpoints')
CONFIG_PATH = os.path.join(BASE_PATH, 'config')

PRED_F = 'generated.txt'

UNK_TOK = '<unk>'
PAD_TOK = '<pad>'
BOS_TOK = '<bos>'
EOS_TOK = '<eos>'
SEP_TOK = '<sep>'

FEAT_COL = 'feature'
ADJ_COL = 'adj'
SCO_COL = 'sco'
REV_COL = 'text'
U_COL = 'user'
I_COL = 'item'
RAT_COL = 'rating'
TIME_COL = 'timestamp'
HIST_I_COL = 'hist_' + I_COL
HIST_FEAT_COL = 'hist_' + FEAT_COL
HIST_ADJ_COL = 'hist_' + ADJ_COL
HIST_REV_COL = 'hist_' + REV_COL
HIST_RAT_COL = 'hist_' + RAT_COL
HIST_REVID_COL = 'hist_rev_id'

HIST_I_IDX_COL = 'hist_'+I_COL+'_index'

RNG_SEED = 1111

HIST_I_MODE = 1
HIST_REV_MODE = 3

HIST_LEN = 20
TXT_LEN = 15

METRICS = ['RMSE↓', 'MAE↓','DIV↓', 'USR↑', 'USN↑', 'FCR↑', 'FMR↑', 'BLEU_1↑', 'BLEU_4↑', 
            'R1_F↑', 'R1_P↑','R1_R↑', 'R2_F↑', 'R2_P↑', 'R2_R↑', 'RL_F↑','RL_P↑', 'RL_R↑']
COLUMN_NAME = ['Model','Dataset', 'Split_ID', 'TestTime','Seed', 
            'RMSE↓', 'MAE↓', 'DIV↓', 'USR↑', 'USN↑', 'FCR↑', 'FMR↑','BLEU_1↑', 'BLEU_4↑',
            'R1_F↑', 'R1_P↑','R1_R↑', 'R2_F↑', 'R2_P↑', 'R2_R↑', 'RL_F↑','RL_P↑', 'RL_R↑',
            'HistLen', 'TextLen','Batch_size','Optimizer','LR',
            'Total_Epoch','Train_Time','Epoch_Time', 'Num_GPU','GPU_ID']
LOG_WARN = 20
LOG_DEBUG_DETAIL = 15
LOG_DEBUG = 10
LOG_INFO = 1
LOG_LEVEL = LOG_INFO