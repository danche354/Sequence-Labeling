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
sys.path.append('../tools')

import train

from tools import conf
from tools import load_data
from tools import prepare
from tools import plot

np.random.seed(0)

# train hyperparameters
feature_length = conf.feature_length
pos_length = conf.pos_length
split_rate = conf.split_rate
batch_size = conf.batch_size
nb_epoch = conf.nb_epoch

model_name = os.path.basename(__file__)[:-3]

folder_path = 'model/%s'%model_name
if not os.path.isdir(folder_path):
    os.makedirs(folder_path)

# the data, shuffled and split between train and test sets
(train_data, dev_data, test_data) = load_data.load_chunk(amount=0, split_rate=split_rate, chunk_type='NP')

train_samples = len(train_data)
dev_samples = len(dev_data)
print('train shape:', train_samples)
print('dev shape:', dev_samples)
print()


senna_input = Input(shape=(step_length,))
senna_embedding = Embedding(130002, 50, weights=[word_embedding], mask_zero=True, input_length=step_length)(query_input)
pos_input = Input(shape=(step_length, pos_length))
senna_pos_merge = merge([senna_embedding, pos_input], mode='concat')
input_mask = Masking(mask_value=0)(senna_pos_merge)
hidden_1 = Bidirectional(LSTM(64, return_sequences=True))(input_mask)
dp_1 = Dropout(0.2)(hidden_1)
hidden_2 = Bidirectional(LSTM(32, return_sequences=True))(dp_1)
dp_2 = Dropout(0.2)(hidden_2)
output = TimeDistributed(Dense(3, activation='softmax'))(dp_2)
model = Model(input=[senna_input,pos_input], output=output)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())


number_of_train_batches = int(math.ceil(float(train_samples)/batch_size))
number_of_test_batches = int(math.ceil(float(dev_samples)/batch_size))


print('start train %s ...\n'%model_name)

accuracy = []
min_loss = 1000
best_epoch = 0

log = open('%s/model_log.txt'%folder_path, 'w')

all_train_loss = []
all_test_loss = []

start_time = datetime.now()
print('train start at %s\n'%str(start_time))
log.write('train start at %s\n\n'%str(start_time))
for epoch in range(nb_epoch):

    start = datetime.now()

    print('-'*60)
    print('epoch %d start at %s'%(epoch, str(start)))

    log.write('-'*60+'\n')
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
        test_metrics = model.test_on_batch(X_test_batch, X_test_batch)
        test_loss += test_metrics[0]
    all_test_loss.append(test_loss)

    if test_loss<min_loss:
        min_loss = test_loss
        best_epoch = epoch

    end = datetime.now()

    model.save('%s/model_epoch_%d.h5'%(folder_path, epoch), overwrite=True)
    auto_encoder.save('%s/hidden_model_epoch_%d.h5'%(folder_path, epoch), overwrite=True)

    print('epoch %d end at %s'%(epoch, str(end)))
    print('epoch %d train loss: %f'%(epoch, train_loss))
    print('epoch %d test loss: %f'%(epoch, test_loss))
    print('best epoch now: %d\n'%best_epoch)

    log.write('epoch %d end at %s\n'%(epoch, str(end)))
    log.write('epoch %d train loss: %f\n'%(epoch, train_loss))
    log.write('epoch %d test loss: %f\n\n'%(epoch, test_loss))

end_time = datetime.now()
print('train end at %s\n'%str(end_time))
log.write('train end at %s\n\n'%str(end_time))

timedelta = end_time - start_time
print('train cost time: %s\n'%str(timedelta))
print('best epoch last: %d\n'%best_epoch)

log.write('train cost time: %s\n\n'%str(timedelta))
log.write('best epoch last: %d\n\n'%best_epoch)

plot.plot(all_train_loss, all_test_loss, title='auto encoder loss', x_lable='epoch', y_label='loss', folder_path=folder_path)

