from keras.models import Model
from keras.layers import Input, Masking, Dense, LSTM
from keras.layers import Dropout, merge, Bidirectional
from keras.layers.embeddings import Embedding
from keras.utils import np_utils

import numpy as np

import sys
import math
import os

# add path
sys.path.append('../')

import train

from tools import conf
from tools import load_data
from tools import prepare_chunk

np.random.seed(0)


print('start train %s ...'%model_name)
train.auto_encoder_start(model=model,
            model_name=model_name,
            nb_epoch=nb_epoch,
            X_train=word_train_data,
            number_of_train_batches=number_of_train_batches,
            X_text=word_test_data,
            number_of_test_batches=number_of_test_batches)

# train hyperparameters
feature_length = conf.feature_length
pos_length = conf.pos_length
split_rate = conf.split_rate
batch_size = conf.word_batch_size
nb_epoch = conf.nb_epoch

def auto_encoder_start(model=model,
            model_name=model_name,
            nb_epoch=nb_epoch,
            X_train=X_train,
            number_of_train_batches=number_of_train_batches,
            X_text=X_test,
            number_of_test_batches=number_of_test_batches):

    folder_path = 'model/chunk/%s'%model_name
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    accuracy = []
    min_loss = 1000
    best_epoch = 0

    log = open('folder_path/model_log.txt', 'w')

    all_train_loss = []
    all_test_loss = []

    for epoch in range(nb_epoch):
        train_loss = 0
        test_loss = 0

        np.random.shuffle(X_train)

        for i in range(number_of_train_batches):
            train_batch = X_train[i*batch_size: (i+1)*batch_size]
            X_train_batch = prepare_data.prepare_chunk(batch=train_batch)
            X_train_batch = X_train_batch.toarray()
            train_metrics = model.train_on_batch(X_train_batch, X_train_batch)
            train_loss += train_metrics[0]
        all_train_loss.append(train_loss)

        for j in range(number_of_test_batches):
            test_batch = X_test[j*batch_size: (j+1)*batch_size]
            X_test_batch = prepare_data.prepare_chunk(batch=train_batch)
            X_test_batch = X_test_batch.toarray()
            test_metrics = model.test_on_batch(X_test_batch)
            test_loss += test_metrics[0]
        all_test_loss.append(test_loss)

        if test_loss<min_loss:
            min_loss = all_error
            best_epoch = epoch

        print('epoch %d train over!'%epoch)
        model.save('folder_path/model_epoch_%d.h5'%epoch, overwrite=True)
        model_2_hidden.save('folder_path/hidden_model_epoch_%d.h5'%epoch, overwrite=True)

