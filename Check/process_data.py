import os
import json
import time
import pandas as pd
import numpy as np

from constants import *


# Download Amazon datasets: wget https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/{Category}.json.gz
# Download Amazon metadata: https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz

def add_ts_amazon():
    new_f = 'reviews_Movies_and_TV_5.json'
    dataset = 'amazon-movies'
    max_seq_len = 20

    # Read original and PETER data
    print('### READING DATA ###')
    start_time = time.time()
    all_data = pd.read_json(os.path.join(DATA_PATHS[dataset], new_f), lines=True, orient='records')
    all_data.rename({'reviewerID': U_COL, 'asin': I_COL, 'unixReviewTime': TIME_COL}, axis=1, inplace=True)
    peter_data = pd.DataFrame.from_records(pd.read_pickle(os.path.join(DATA_PATHS[dataset], 'reviews.pickle')))
    print(f'Original data has {all_data.shape[0]} reviews while PETER data has {peter_data.shape[0]} records.')
    print(f'Original fields are: {all_data.columns.values.tolist()}')
    print(f'ORIG: Max. count of user-item pairs: {all_data[[U_COL, I_COL]].value_counts().max()}')
    print(f'ORIG: Max. count of user review: {all_data[U_COL].value_counts().max()}')
    print(f'PETER: Max. count of user-item pairs: {peter_data[[U_COL, I_COL]].value_counts().max()}')
    print(f'PETER: Max. count of user review: {peter_data[U_COL].value_counts().max()}')
    print(f'Ellapsed time: {time.time() - start_time:.2f}s')

    # peter_data = peter_data[peter_data[U_COL].isin(peter_data[U_COL].value_counts()[:5].index.values)]

    # Unpack PETER template column
    print('### SPLITTING TEMPLATE FEATURE ###')
    start_time = time.time()
    peter_data.reset_index(drop=True, inplace=True)
    peter_data = pd.concat([peter_data, pd.DataFrame(peter_data['template'].tolist(),
                                                     columns=[FEAT_COL, ADJ_COL, REV_COL, SCO_COL])],
                           axis=1)
    peter_data.drop('template', axis=1, inplace=True)
    print(f'Ellapsed time: {time.time() - start_time:.2f}s')
    
    # Filter user-item pairs that appear in peter_data (at least 20 interactions per user)
    # peter_data['ui'] = peter_data[U_COL] + '-' + peter_data[I_COL]
    # all_data['ui'] = all_data[U_COL] + '-' + all_data[I_COL]
    # all_data = all_data[all_data['ui'].isin(peter_data['ui'].unique())]
    # peter_data.drop('ui', axis=1, inplace=True)
    # all_data.drop('ui', axis=1, inplace=True)

    # Add history features to original dataset
    print('### ADDING TIMESTAMPS TO PETER DATA ###')
    start_time = time.time()
    peter_data['ui'] = peter_data[U_COL] + '-' + peter_data[I_COL]
    all_data['ui'] = all_data[U_COL] + '-' + all_data[I_COL]
    all_data.set_index('ui', drop=True, inplace=True)
    print(f'NaN values in PETER user-item combination: {peter_data["ui"].isna().sum()}')
    print(f'PETER UI max count: {peter_data["ui"].value_counts().max()}')
    peter_data[TIME_COL] = all_data.loc[peter_data['ui'].values, TIME_COL].values
    peter_data.drop('ui', axis=1, inplace=True)
    del all_data
    print(f'Ellapsed time: {time.time() - start_time:.2f}s')
    print(f'Max user interactions at the same timestamp: {peter_data.value_counts(subset=[U_COL, TIME_COL]).max()}')

    print('### ADDING HISTORY FEATURES TO PETER DATA ###')
    start_time = time.time()
    hist_df = []
    peter_data.sort_values(TIME_COL, inplace=True)
    for uid, udata in peter_data.groupby(U_COL):
        for t in range(udata.shape[0]):
            record = udata.iloc[t].values.tolist()
            record += udata.iloc[max(0, t-max_seq_len):t][[I_COL, FEAT_COL, ADJ_COL, REV_COL, RAT_COL]].values.T.tolist()
            hist_df.append(record)
    peter_data = pd.DataFrame(hist_df, columns=peter_data.columns.tolist() +
                                               [HIST_I_COL, HIST_FEAT_COL, HIST_ADJ_COL, HIST_REV_COL, HIST_RAT_COL])
    print(f'Ellapsed time: {time.time() - start_time:.2f}s')
    print(f'PETER NaNs: {peter_data.isna().sum().sum()}')
    print(f'PETER records: {peter_data.shape[0]}')
    print(f'PETER columns: {peter_data.columns.tolist()}')

    pd.to_pickle(peter_data.to_dict(orient='records'), os.path.join(DATA_PATHS[dataset], 'reviews_new.pickle'))

    print('Finished')


def comparison_amazon():
    new_f = 'reviews_Movies_and_TV_5.json'
    dataset = 'amazon_movies'

    # Read original and PETER data
    print('### READING DATA ###')
    start_time = time.time()
    all_data = pd.read_json(os.path.join(DATA_PATHS[dataset], new_f), lines=True, orient='records')
    all_data.rename({'reviewerID': U_COL, 'asin': I_COL}, axis=1, inplace=True)
    peter_data = pd.DataFrame.from_records(pd.read_pickle(os.path.join(DATA_PATHS[dataset], 'reviews.pickle')))
    peter_data['template'] = peter_data['template'].apply(lambda s: s[2])
    print(f'Original data has {all_data.shape[0]} reviews while PETER data has {peter_data.shape[0]} records.')
    print(f'ORIG: Max. count of user-item pairs: {all_data[[U_COL, I_COL]].value_counts().max()}')
    print(f'ORIG: Max. count of user review: {all_data[U_COL].value_counts().max()}')
    print(f'PETER: Max. count of user-item pairs: {peter_data[[U_COL, I_COL]].value_counts().max()}')
    print(f'PETER: Max. count of user review: {peter_data[U_COL].value_counts().max()}')
    print(f'Ellapsed time: {time.time() - start_time:.2f}s')

    # Check whether all PETER user-item combinations are in Amazon dataset
    print('### CHECK 1 ###')
    start_time = time.time()
    peter_data['ui'] = peter_data[U_COL] + '-' + peter_data[I_COL]
    all_data['ui'] = all_data[U_COL] + '-' + all_data[I_COL]
    try:
        assert len(set(peter_data['ui']).intersection(all_data['ui'].unique())) == peter_data['ui'].nunique()
    except AssertionError:
        print('Check Failed!')
    print(f'Ellapsed time: {time.time() - start_time:.2f}s')

    # Check whether for all PETER users, we have all Amazon reviews
    print('### CHECK 2 ###')
    start_time = time.time()
    all_data = all_data[all_data[U_COL].isin(peter_data[U_COL].unique())]
    try:
        assert len(set(all_data['ui']).intersection(peter_data['ui'].unique())) == all_data['ui'].nunique()
    except AssertionError:
        print('Check Failed!')
    print(f'Ellapsed time: {time.time() - start_time:.2f}s')


def split_data(test_ratio=0.1):
    val_ratio = test_ratio

    dataset = 'amazon-movies'
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


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', None)
    pd.set_option('expand_frame_repr', False)
    # add_ts_amazon()
    split_data(test_ratio=0.1)
