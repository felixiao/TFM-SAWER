import os
import re
import pandas as pd

from constants import *


def check_model(model_name):
    log_path = os.path.join(BASE_PATH, 'logs')
    res = []
    for dataset in DATA_PATHS:
        files = [f for f in os.listdir(log_path) if re.match(f'{dataset}_{model_name}\d+\.txt', f)]
        for f in files:
            read_results(log_path, f, res, dataset, model_name)
    res = pd.DataFrame.from_records(res)
    cols = ['BLEU-1', 'BLEU-4', 'USR', 'USN', 'DIV', 'FCR', 'FMR', 'rouge_1/f_score', 'rouge_1/r_score',
            'rouge_1/p_score', 'rouge_2/f_score', 'rouge_2/r_score', 'rouge_2/p_score']
    print(res.groupby('dataset').mean()[cols].round(2).to_csv(index=False))


def check_split(fold=0):
    log_path = os.path.join(BASE_PATH, 'logs')
    res = []
    files = []
    for dataset in ['amazon-movies']:
        for f in os.listdir(log_path):
            if f.startswith(dataset) and str(fold) in f and f != 'amazon-movies_sequer-modrat_0_1.txt':
                _, model_name, split_ix, run_id = f[:-len('.txt')].split('_')
                # model_name, split_ix = re.findall(f'{dataset}_(\w+)(\d+)\.txt', f)[0]
                if int(split_ix) == fold:
                    read_results(log_path, f, res, dataset, model_name, run_id)

    res = pd.DataFrame.from_records(res)
    cols = ['model_name'] + METRICS

    models = [m for m in res['model_name'].unique() if m != 'peter']
    res = res[cols].groupby('model_name').mean()
    for model in models:
        res.loc[f'improv_{model}(%)'] = (res.loc[model] - res.loc['peter']) / res.loc['peter'] * 100
    print(res.round(2).to_csv())


def read_results(log_path, f, res, dataset, model_name, run_id=0):
    res.append({'dataset': dataset, 'model_name': model_name, 'run_id': run_id})
    with open(os.path.join(log_path, f), 'r') as f:
        log = f.read()
    results = re.split('=+', log)[-1].replace('\n', ' ')
    for m in METRICS:
        match = re.findall(f'{m}\s+(\d+(\.\d+)?)', results)
        if match:
            res[-1][m] = float(match[0][0])


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', None)
    pd.set_option('expand_frame_repr', False)
    # check_model('peter')
    check_split(0)
