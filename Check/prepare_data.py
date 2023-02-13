import pandas as pd
import numpy as np

datasetname = 'SportsAndOutdoors'
print('Dataset Name:',datasetname)
reviews = pd.DataFrame.from_records(pd.read_pickle(f'./Data/Amazon/{datasetname}/reviews.pickle'))
print('Cols',list(reviews.columns))





print(reviews[:2])
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


# print('Dataset Name: MoviesAndTV_New')
# mov_reviews = pd.DataFrame.from_records(pd.read_pickle(f'./Data/Amazon/MoviesAndTV_New/Movie_and_TV_index.pickle'))
# print('Cols',list(mov_reviews.columns))
# print(mov_reviews[:2])