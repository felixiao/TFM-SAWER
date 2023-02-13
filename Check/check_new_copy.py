import pandas as pd
import numpy as np
import pickle
import os
import time
from tqdm import tqdm
# datasetname = 'MoviesAndTV_New'
# print('Dataset Name:',datasetname)
# with open(f'./Data/Amazon/{datasetname}/reviews_new.pickle', "rb") as f:
#     reviews = pickle.load(f)

# reviews = reviews.to_dict('records')
# # print('Cols',list(reviews.columns))
# print(reviews[:2])

# with open(f'./Data/Amazon/{datasetname}/reviews_new.pickle', "wb") as f:
#     reviews = pickle.dump(reviews,f)

BASE_PATH = './P5'
DATA_PATHS = {'amazon-sports':'./Data/Amazon/SportsAndOutdoors',
                'amazon-toys':'./Data/Amazon/ToysAndGames',
                'amazon-beauty':'./Data/Amazon/Beauty'}
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

def amazon_p5_to_peter():
    # import pickle5 as pickle

    for folder in ['sports', 'beauty', 'toys']:
        print(f'Processing {folder}...')
        with open(os.path.join(BASE_PATH, 'data', folder, 'review_splits-v4.pkl'), 'rb') as f:
            data = pickle.load(f)

        data = data['train'] + data['val'] + data['test']
        print(f'Orig. number of records: {len(data)}')

        # exp_splits.pkl are supposed to have a feature, but just in case
        # data = [r for r in data if 'feature' in r]
        # print(f'Number of records with feature: {len(data)}')

        data = pd.DataFrame.from_records(data)
        column_map = {'reviewerID': U_COL, 'asin': I_COL, 'reviewText': REV_COL, 'unixReviewTime': TIME_COL,
                      'overall': RAT_COL, 'feature': FEAT_COL}
        data.rename(column_map, axis=1, inplace=True)
        data = data[column_map.values()]
        data[FEAT_COL] = data[FEAT_COL].fillna('')
        data[ADJ_COL] = ''

        # Don't filter P5 datasets with k-core
        # data = iterative_kcore_filter(data, kcore=5, verbose=1)
        # data = data[~(data[FEAT_COL] == '')]

        # Add sequential fields
        data = add_hist_feats(data, max_seq_len=20)
        if not os.path.exists(DATA_PATHS['amazon-' + folder]):
            os.mkdir(DATA_PATHS['amazon-' + folder])
        pd.to_pickle(data.to_dict(orient='records'), os.path.join(DATA_PATHS['amazon-' + folder], 'reviews_new.pickle'))

    print('Finished')

def add_hist_feats(data, max_seq_len=20):
    print('### ADDING HISTORY FEATURES TO PETER DATA ###')
    start_time = time.time()
    hist_df = []
    data.sort_values(TIME_COL, inplace=True)
    for uid, udata in tqdm(data.groupby(U_COL)):
        for t in range(udata.shape[0]):
            record = udata.iloc[t].values.tolist()
            record += udata.iloc[max(0, t - max_seq_len):t][[I_COL, FEAT_COL, ADJ_COL, REV_COL, RAT_COL]].values.T.tolist()
            hist_df.append(record)
    data = pd.DataFrame(hist_df, columns=data.columns.tolist() +
                                               [HIST_I_COL, HIST_FEAT_COL, HIST_ADJ_COL, HIST_REV_COL, HIST_RAT_COL])
    print(f'Ellapsed time: {time.time() - start_time:.2f}s')
    print(f'Existing NaNs: {data.isna().sum().sum()}')
    print(f'Total records: {data.shape[0]}')
    print(f'Columns: {data.columns.tolist()}')
    return data


def split_data(test_ratio=0.1):
    val_ratio = test_ratio

    dataset = 'amazon-sports'
    peter_data = pd.read_pickle(os.path.join(DATA_PATHS[dataset], 'reviews_new.pickle'))
    peter_data = pd.DataFrame.from_records(peter_data)

    peter_data.sort_values(by=[U_COL, TIME_COL], inplace=True)
    ucounts = peter_data[U_COL].value_counts().values
    uoffsets = ucounts.cumsum()
    split_ixs = np.zeros((peter_data.shape[0], ), dtype=int)
    if isinstance(test_ratio, float):
        assert isinstance(val_ratio, float)
        assert test_ratio < 1.0
        tst_start_ixs = uoffsets - (ucounts * test_ratio).astype(int)
        val_start_ixs = tst_start_ixs - (ucounts * val_ratio).astype(int)
    elif isinstance(test_ratio, int):
        assert isinstance(val_ratio, int)
        assert all(ucounts > (test_ratio + val_ratio))
        tst_start_ixs = uoffsets - test_ratio
        val_start_ixs = tst_start_ixs - val_ratio
    else:
        raise TypeError('test_ratio is neither int nor float')
    for vix, tix, offset in zip(val_start_ixs, tst_start_ixs, uoffsets):
        split_ixs[tix:offset] = 2
        split_ixs[vix:tix] = 1

    np.savetxt(os.path.join(DATA_PATHS[dataset], '0', 'train.index'),
               peter_data.index.values[split_ixs == 0],
               delimiter=' ', fmt="%d")

    np.savetxt(os.path.join(DATA_PATHS[dataset], '0', 'validation.index'),
               peter_data.index.values[split_ixs == 1],
               delimiter=' ', fmt="%d")

    np.savetxt(os.path.join(DATA_PATHS[dataset], '0', 'test.index'),
               peter_data.index.values[split_ixs == 2],
               delimiter=' ', fmt="%d")

    print('Finished!')

# amazon_p5_to_peter()

split_data()

# datasetname = 'beauty'
# print('Dataset Name:',datasetname)
# with open(f'./Data/Amazon/{datasetname}/reviews_new.pickle', "rb") as f:
#     reviews = pickle.load(f)

# reviews = reviews.to_dict('records')
# # print('Cols',list(reviews.columns))
# print(reviews[:2])

# with open(f'./Data/Amazon/{datasetname}/reviews_new.pickle', "wb") as f:
#     reviews = pickle.dump(reviews,f)