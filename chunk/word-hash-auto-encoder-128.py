from keras.models import Model
from keras.layers import Input, Masking, Dense, LSTM
from keras.layers import Dropout, merge, Bidirectional
from keras.layers.embeddings import Embedding
from keras.utils import np_utils

import numpy as np

import sys
import math
import os
from datetime import datetime

# add path
sys.path.append('../')

from tools import conf
from tools import load_data
from tools import prepare
from tools import plot

np.random.seed(0)

# train hyperparameters
feature_length = conf.feature_length
pos_length = conf.pos_length
split_rate = conf.split_rate
batch_size = conf.word_batch_size
nb_epoch = conf.nb_epoch

model_name = os.path.basename(__file__)[:-3]

if not os.path.isdir('./log/%s'%model_name):
    os.makedirs('./log/%s'%model_name)
if not os.path.isdir('./model/%s'%model_name):
    os.makedirs('./model/%s'%model_name)
if not os.path.isdir('./fig/%s'%model_name):
    os.makedirs('./fig/%s'%model_name)

# the data, shuffled and split between train and test sets
(train_data, dev_data, test_data) = load_data.load_chunk(amount=0, split_rate=split_rate)

train_samples = len(train_data) + len(dev_data)
dev_samples = len(test_data)
print('train shape:', train_samples)
print('test shape:', dev_samples)

word_train_data =[]
word_dev_data = []
# all train sample, combine train and dev
[word_train_data.extend(list(each[0])) for each in train_data]
[word_train_data.extend(list(each[0])) for each in dev_data]
[word_test_data.extend(list(each[0])) for each in test_data]

word_train_samples=len(word_train_data)
word_dev_samples=len(word_test_data)
print('word train shape:', word_train_samples)
print('word test shape:', word_dev_samples)


# model structure
word_input = Input(shape=(feature_length, ))
hidden = Dense(128, activation='tanh')(word_input)
after_dp = Dropout(0.2)(hidden)
word_output = Dense(feature_length)(after_dp)
model = Model(input=word_input, output=word_output)
auto_encoder = Model(input=word_input, output=hidden)
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())


number_of_train_batches = int(math.ceil(float(word_train_samples)/word_batch_size))
number_of_test_batches = int(math.ceil(float(word_test_samples)/word_batch_size))


print('start train auto-encoder ...')


folder_path = 'model/chunk/%s'%model_name
if not os.path.isdir(folder_path):
    os.makedirs(folder_path)

accuracy = []
min_loss = 1000
best_epoch = 0

log = open('folder_path/model_log.txt', 'w')

all_train_loss = []
all_test_loss = []

start_time = datetime.now()
log.write('train start at %s\n'str(start_time))
for epoch in range(nb_epoch):

    start = datetime.now()

    log.write('epoch %d start at %s\n'%(epoch, str(start)))
    train_loss = 0
    test_loss = 0

    np.random.shuffle(word_train_data)

    for i in range(number_of_train_batches):
        train_batch = word_train_data[i*batch_size: (i+1)*batch_size]
        X_train_batch = prepare.prepare_chunk_encoder(batch=train_batch)
        X_train_batch = X_train_batch.toarray()
        train_metrics = model.train_on_batch(X_train_batch, X_train_batch)
        train_loss += train_metrics[0]
    all_train_loss.append(train_loss)

    for j in range(number_of_test_batches):
        test_batch = word_test_data[j*batch_size: (j+1)*batch_size]
        X_test_batch = prepare.prepare_chunk_encoder(batch=train_batch)
        X_test_batch = X_test_batch.toarray()
        test_metrics = model.test_on_batch(X_test_batch)
        test_loss += test_metrics[0]
    all_test_loss.append(test_loss)

    if test_loss<min_loss:
        min_loss = all_error
        best_epoch = epoch

    end = datetime.now()

    model.save('folder_path/model_epoch_%d.h5'%epoch, overwrite=True)
    auto_encoder.save('folder_path/hidden_model_epoch_%d.h5'%epoch, overwrite=True)

    print('-'*60+'\n')
    print('epoch %d train over !\n'%epoch)
    print('epoch %d train loss: %f\n'%(epoch, train_loss))
    print('epoch %d test loss: %f\n'%(epoch, test_loss))
    print('best epoch now: %d\n\n'%epoch)

    log.write('-'*60+'\n')
    log.write('epoch %d train over !\n'%epoch)
    log.write('epoch %d train loss: %f\n'%(epoch, train_loss))
    log.write('epoch %d test loss: %f\n'%(epoch, test_loss))
    log.write('best epoch now: %d\n\n'%epoch)
    log.write('epoch %d end at %s\n'%(epoch, str(end)))

end_time = datetime.now()
log.write('train end at %s\n'str(end_time))

timedelta = end_time - start_time
log.write('train cost time: %s'%str(timedelta))

plot.plot(all_train_loss, all_test_loss, title='auto encoder loss', x_lable='epoch', y_label='loss', folder_path=folder_path)

