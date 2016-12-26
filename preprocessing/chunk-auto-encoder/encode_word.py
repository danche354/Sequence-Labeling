from keras.models import load_model

import pandas as pd
import numpy as np

import sys
import os

# change dir to train file, for environment
os.chdir('../../chunk/')

# add path
sys.path.append('../')
sys.path.append('../tools')

from tools import load_data
from tools import prepare

model_path = './model/word-hash-auto-encoder-128/hidden_model_epoch_27.h5'

w = open('../preprocessing/chunk-auto-encoder/conll2000-word.lst', 'w')
embeddings = pd.DataFrame(columns=range(128))

print('loading model...')
encoder = load_model(model_path)
print('loading model finished.')

train_data, dev_data = load_data.load_chunk(dataset='train.txt', split_rate=0.9)
test_data = load_data.load_chunk(dataset='test.txt')

all_word =[]

# all word
[all_word.extend(list(each[0])) for each in train_data]
[all_word.extend(list(each[0])) for each in dev_data]
[all_word.extend(list(each[0])) for each in test_data]

all_word = [each.strip().lower() for each in all_word]
all_word = list(set(all_word))

for i, word in enumerate(all_word):
    w.write(word+'\n')
    word_hashing = prepare.prepare_auto_encoder(batch=[word], task='chunk')
    word_hashing = word_hashing.toarray()
    representation = encoder.predict_on_batch(word_hashing)
#normalization = (representation-np.mean(representation))/np.std(representation)
    normalization = (representation-np.min(representation))/(np.max(representation)-np.min(representation))
    embeddings.loc[i] = normalization[0]

embeddings.to_csv('../preprocessing/chunk-auto-encoder/auto-encoder-embeddings.txt', sep=' ',header=False,index=False,float_format='%.6f')
w.close()

