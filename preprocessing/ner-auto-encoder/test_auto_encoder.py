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

epoch = sys.argv[1]
test = sys.argv[2]

path = './model/word-hash-auto-encoder-128/model_epoch_%s.h5'%epoch

model = load_model(path)

train_data = load_data.load_chunk(dataset='eng.train')
dev_data = load_data.load_chunk(dataset='eng.testa')
test_data = load_data.load_chunk(dataset='eng.testb')

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

train_word_dict = {}
dev_word_dict = {}
test_word_dict = {}

for each in train_word:
    if each in train_word_dict:
        train_word_dict[each] += 1
    else:
        train_word_dict[each] = 1
    
for each in dev_word:
    if each in dev_word_dict:
        dev_word_dict[each] += 1
    else:
        dev_word_dict[each] = 1

for each in test_word:
    if each in test_word_dict:
        test_word_dict[each] += 1
    else:
        test_word_dict[each] = 1

train_word = train_word_dict.keys()
dev_word = dev_word_dict.keys()
test_word = test_word_dict.keys()

if test=='dev':
    word = dev_word[:20]
elif test=='test':
    word = test_word[:20]
else:
    word = train_word[:20]

word_hashing = prepare.prepare_auto_encoder(batch=word,  task='ner')
word_hashing = word_hashing.toarray()
output = model.predict_on_batch(word_hashing)

while True:
    number = input('please input word index: ')
    exist = word[number]
    print('word is: ' + exist)
    if exist in train_word_dict:
        print('    in train: ' + str(train_word_dict[exist]) + ' times.')
    if exist in dev_word_dict:
        print('    in dev: ' + str(dev_word_dict[exist]) + ' times.')
    if exist in test_word_dict:
        print('    in test: ' + str(test_word_dict[exist]) + ' times.')
    print('-'*60)
    ind = []
    for i, e in enumerate(word_hashing[number]):
        if e==1:
            print(i)
            ind.append(i)
    print('word_hasing'+ '-'*60)
    
    for i in ind:
        print(output[number][i])
    print('output'+ '-'*60)
