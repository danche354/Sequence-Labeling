from keras.models import load_model

import pandas as pd
import numpy as np

import sys
import os

# change dir to train file, for environment
os.chdir('../../ner/')

# add path
sys.path.append('../')
sys.path.append('../tools')

from tools import load_data
from tools import prepare

model_path = './model/word-hash-2-auto-encoder-128/hidden_model_epoch_22.h5'

w = open('../preprocessing/ner-auto-encoder-2/conll2003-word.lst', 'w')
embeddings = pd.DataFrame(columns=range(128))

print('loading model...')
encoder = load_model(model_path)
print('loading model finished.')

train_data = load_data.load_ner(dataset='eng.train')
dev_data = load_data.load_ner(dataset='eng.testa')
test_data = load_data.load_ner(dataset='eng.testb')

all_word =[]

# all word
[all_word.extend(list(each[0])) for each in train_data]
[all_word.extend(list(each[0])) for each in dev_data]
[all_word.extend(list(each[0])) for each in test_data]

all_word = [each.strip().lower() for each in all_word]
all_word = list(set(all_word))

for i, word in enumerate(all_word):
    w.write(word+'\n')
    word_hashing = prepare.prepare_auto_encoder(batch=[word], task='ner', gram='bi')
    word_hashing = word_hashing.toarray()
    representation = encoder.predict_on_batch(word_hashing)
    #normalization = (representation-np.mean(representation))/np.std(representation)
    normalization = (representation-np.min(representation))/(np.max(representation)-np.min(representation))
    embeddings.loc[i] = normalization[0]

embeddings.to_csv('../preprocessing/ner-auto-encoder-2/auto-encoder-embeddings.txt', sep=' ',header=False,index=False,float_format='%.6f')
w.close()

