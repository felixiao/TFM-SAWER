import pandas as pd
import numpy as np


datasetname = 'MusicalInstruments'
print('Dataset Name:',datasetname)
reviews = pd.DataFrame.from_records(pd.read_pickle(f'./Data/Amazon/{datasetname}/reviews.pickle'))


# datasetname = 'SportsAndOutdoors'
# print('Dataset Name:',datasetname)
# reviews = pd.DataFrame.from_records(pd.read_pickle(f'./Data/Amazon/{datasetname}/reviews_new.pickle'))

print('Cols',list(reviews.columns))

print('Sent',reviews['sentence'][:5])

# print('template',reviews['template'][:3])
# print('predicted',reviews['predicted'][:3])
# print('#Records',len(reviews))            
# print('#Feat',len(reviews['feature'].unique()))
# print('#Adj',len(reviews['adj'].unique()))
# words = [len(s.split(' ')) for s in reviews['text']]
# hists = [len(h) for h in reviews['hist_item']]

# hists_len = np.sum(hists)

# words_len = np.sum(words)

# print('#Sent',len(reviews['text']))
# print('#Word, ratio',words_len,words_len/len(reviews['text']))
# print('#Hist, ratio',hists_len,hists_len/len(reviews['hist_item']))

# print('#user, ratio',len(reviews['user'].unique()), len(reviews['user'])/len(reviews['user'].unique()))
# print('#item, ratio',len(reviews['item'].unique()),len(reviews['item'])/len(reviews['item'].unique()))