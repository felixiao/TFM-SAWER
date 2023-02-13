import pandas as pd
import numpy as np
datasetname = 'Sports And Outdoors'
print('Dataset Name:',datasetname)
reviews = pd.DataFrame.from_records(pd.read_pickle(f'./Data/Amazon/SportsAndOutdoors/reviews_new.pickle'))

print('Cols',list(reviews.columns))
print('#records',len(reviews))
attr = []
feat = []
sent = []
word = []
for i in reviews.index:
    # print(str(i), reviews['user'][i],reviews['sentence'][i])
    # print(type(reviews['sentence'][i]))
    if type(reviews['sentence'][i]) == type([]):
        for s in reviews['sentence'][i]:
            # print(i,j,s)
            attr.append(s[0])
            feat.append(s[1])
            sent.append(s[2])
            for w in s[2].split(' '):
                word.append(w)
print('#Attr',len(attr))
print('#Feat',len(feat))
print('#Sent',len(sent))
print('#Word, ratio',len(word),len(word)/len(sent))
print('Word:',word[:50])
print('Attr',attr[10:15])
print('Feat',feat[10:15])

print('#attr',len(set(attr)))
print('#feat',len(set(feat)))


print('#user, ratio',len(reviews['user'].unique()), len(reviews['user'])/len(reviews['user'].unique()))
print('#item, ratio',len(reviews['item'].unique()),len(reviews['item'])/len(reviews['item'].unique()))

# print('unique user idx/ total',len(reviews['user_index'].unique()),len(reviews['user_index']))
# print('unique item idx/ total',len(reviews['item_index'].unique()),len(reviews['item_index']))

# print('text',reviews['text'][12])
# print('sent',reviews['sentence'][12])
# len_sents = [len(reviews['sentence'][i][:][2].split(' ')) for i in reviews.index]

# print('#words', np.sum(len_sents))