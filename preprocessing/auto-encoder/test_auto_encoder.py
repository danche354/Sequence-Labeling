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

epoch = sys.argv[1]
test = sys.argv[2]

path = './model/word-hash-auto-encoder-128/hidden_model_epoch_%s.h5'%epoch

model = load_data(path)

train_data, dev_data = load_data.load_chunk(dataset='train.txt', split_rate=0.9)
test_data = load_data.load_chunk(dataset='test.txt')

train_word = []
dev_word = []
test_word = []

# all word
[train_word.extend(list(each[0])) for each in train_data]
[dev_word.extend(list(each[0])) for each in dev_data]
[test_word.extend(list(each[0])) for each in test_data]

train_word = [each.strip().lower() for each in train_word]
dev_word = [each.strip().lower() for each in dev_word]
test_word = [each.strip().lower() for each in test_word]

train_word = list(set(train_word))
dev_word = list(set(dev_word))
test_word = list(set(test_word))

if test=='dev':
    word = dev_word[:20]
elif test='test':
    word = test_word[:20]
else:
    word = train_word[:20]

word_hashing = prepare.prepare_chunk_encoder(batch=word)
word_hashing = word_hashing.toarray()
output = model.predict_on_batch(word_hashing)

while True:
    number = input('please input word index: ')
    for i, e in word_hashing[number]:
        if e==1:
            print(i)
    print('word_hasing'+ '-'*60)

    threshold = input('please input threshold: ')
    for i, e in output[number]:
        if e>=threshold:
            print(i)
    print('output'+ '-'*60)