from keras.models import Model
from keras.layers import Input, Masking, Dense, LSTM
from keras.layers import Dropout, merge, Bidirectional
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.optimizers import Adam

import numpy as np

import sys
import math
import os
from datetime import datetime

# add path
sys.path.append('../')
sys.path.append('../tools')

from tools import conf
from tools import load_data
from tools import prepare
from tools import plot

np.random.seed(0)

# train hyperparameters
feature_length = conf.chunk_feature_length

split_rate = conf.chunk_split_rate
batch_size = conf.word_batch_size
nb_epoch = conf.word_nb_epoch

model_name = os.path.basename(__file__)[:-3]

folder_path = 'model/%s'%model_name
if not os.path.isdir(folder_path):
    os.makedirs(folder_path)

# the data, shuffled and split between train and dev sets
train_data, dev_data = load_data.load_chunk(dataset='train.txt', split_rate=split_rate)

train_samples = len(train_data)
dev_samples = len(dev_data)
print('train shape:', train_samples)
print('dev shape:', dev_samples)
print()

word_train_data =[]
word_dev_data = []
# all train sample, combine train and dev
[word_train_data.extend(list(each[0])) for each in train_data]
[word_dev_data.extend(list(each[0])) for each in dev_data]

word_train_samples=len(word_train_data)
word_dev_samples=len(word_dev_data)
print('word train shape:', word_train_samples)
print('word dev shape:', word_dev_samples)

adam = Adam(lr=0.0001)

# model structure
word_input = Input(shape=(feature_length, ))
hidden = Dense(128)(word_input)
dp = Dropout(0.5)(hidden)
word_output = Dense(feature_length)(dp)
model = Model(input=word_input, output=word_output)
auto_encoder = Model(input=word_input, output=hidden)
model.compile(loss='mse',
              optimizer=adam,
              metrics=['accuracy'])

print(model.summary())


number_of_train_batches = int(math.ceil(float(word_train_samples)/batch_size))
number_of_dev_batches = int(math.ceil(float(word_dev_samples)/batch_size))


print('start train %s ...\n'%model_name)

min_loss = 1000
best_epoch = 0
all_train_loss = []
all_dev_loss = []

log = open('%s/model_log.txt'%folder_path, 'w')

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
    dev_loss = 0

    np.random.shuffle(word_train_data)

    for i in range(number_of_train_batches):
        train_batch = word_train_data[i*batch_size: (i+1)*batch_size]
        X_train_batch = prepare.prepare_auto_encoder(batch=train_batch, task='chunk')
        X_train_batch = X_train_batch.toarray()
        train_metrics = model.train_on_batch(X_train_batch, X_train_batch)
        train_loss += train_metrics[0]
    all_train_loss.append(train_loss)

    for j in range(number_of_dev_batches):
        dev_batch = word_dev_data[j*batch_size: (j+1)*batch_size]
        X_dev_batch = prepare.prepare_auto_encoder(batch=dev_batch, task='chunk')
        X_dev_batch = X_dev_batch.toarray()
        dev_metrics = model.test_on_batch(X_dev_batch, X_dev_batch)
        dev_loss += dev_metrics[0]
    all_dev_loss.append(dev_loss)

    if dev_loss<min_loss:
        min_loss = dev_loss
        best_epoch = epoch

    end = datetime.now()

    model.save('%s/model_epoch_%d.h5'%(folder_path, epoch), overwrite=True)
    auto_encoder.save('%s/hidden_model_epoch_%d.h5'%(folder_path, epoch), overwrite=True)

    print('epoch %d end at %s'%(epoch, str(end)))
    print('epoch %d train loss: %f'%(epoch, train_loss))
    print('epoch %d dev loss: %f'%(epoch, dev_loss))
    print('best epoch now: %d\n'%best_epoch)

    log.write('epoch %d end at %s\n'%(epoch, str(end)))
    log.write('epoch %d train loss: %f\n'%(epoch, train_loss))
    log.write('epoch %d dev loss: %f\n\n'%(epoch, dev_loss))

end_time = datetime.now()
print('train end at %s\n'%str(end_time))
log.write('train end at %s\n\n'%str(end_time))

timedelta = end_time - start_time
print('train cost time: %s\n'%str(timedelta))
print('best epoch last: %d\n'%best_epoch)

log.write('train cost time: %s\n\n'%str(timedelta))
log.write('best epoch last: %d\n\n'%best_epoch)

plot.plot_loss(all_train_loss, all_dev_loss, folder_path=folder_path, title='%s'%model_name)

