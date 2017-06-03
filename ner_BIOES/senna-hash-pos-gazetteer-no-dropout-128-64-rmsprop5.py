from keras.models import Model
from keras.layers import Input, Masking, Dense, LSTM
from keras.layers import Dropout, TimeDistributed, Bidirectional, merge
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.optimizers import RMSprop

import numpy as np
import pandas as pd

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
step_length = conf.ner_step_length
pos_length = conf.ner_pos_length
chunk_length = conf.ner_chunk_length
gazetteer_length = conf.gazetteer_length

emb_vocab = conf.senna_vocab
emb_length = conf.senna_length

hash_vocab = conf.ner_hash_vocab
hash_length = conf.ner_hash_length

output_length = conf.ner_BIOES_length

batch_size = conf.batch_size
nb_epoch = 70 #conf.nb_epoch

model_name = os.path.basename(__file__)[:-3]

folder_path = 'model/%s'%model_name
if not os.path.isdir(folder_path):
    os.makedirs(folder_path)

# the data, shuffled and split between train and test sets
train_data = load_data.load_ner(dataset='eng.train', form='BIOES')
dev_data = load_data.load_ner(dataset='eng.testa', form='BIOES')

train_samples = len(train_data)
dev_samples = len(dev_data)
print('train shape:', train_samples)
print('dev shape:', dev_samples)
print()

word_embedding = pd.read_csv('../preprocessing/senna/embeddings.txt', delimiter=' ', header=None)
word_embedding = word_embedding.values
word_embedding = np.concatenate([np.zeros((1,emb_length)),word_embedding, np.random.uniform(-1,1,(1,emb_length))])

hash_embedding = pd.read_csv('../preprocessing/ner-auto-encoder/auto-encoder-embeddings.txt', delimiter=' ', header=None)
hash_embedding = hash_embedding.values
hash_embedding = np.concatenate([np.zeros((1,hash_length)),hash_embedding, np.random.rand(1,hash_length)])

embed_index_input = Input(shape=(step_length,))
embedding = Embedding(emb_vocab+2, emb_length, weights=[word_embedding], mask_zero=True, input_length=step_length)(embed_index_input)

hash_index_input = Input(shape=(step_length,))
encoder_embedding = Embedding(hash_vocab+2, hash_length, weights=[hash_embedding], mask_zero=True, input_length=step_length)(hash_index_input)

pos_input = Input(shape=(step_length, pos_length))
# chunk_input = Input(shape=(step_length, chunk_length))
gazetteer_input = Input(shape=(step_length, gazetteer_length))

senna_hash_pos_chunk_gazetteer_merge = merge([embedding, encoder_embedding, pos_input, gazetteer_input], mode='concat')
input_mask = Masking(mask_value=0)(senna_hash_pos_chunk_gazetteer_merge)
#dp_1 = Dropout(0.5)(input_mask)
hidden_1 = Bidirectional(LSTM(128, return_sequences=True))(input_mask)
hidden_2 = Bidirectional(LSTM(64, return_sequences=True))(hidden_1)
#dp_2 = Dropout(0.5)(hidden_2)
output = TimeDistributed(Dense(output_length, activation='softmax'))(hidden_2)
model = Model(input=[embed_index_input,hash_index_input,pos_input,gazetteer_input], output=output)

rmsprop = RMSprop(lr=0.0005)

model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

print(model.summary())


number_of_train_batches = int(math.ceil(float(train_samples)/batch_size))
number_of_dev_batches = int(math.ceil(float(dev_samples)/batch_size))


print('start train %s ...\n'%model_name)

best_accuracy = 0
best_epoch = 0
all_train_loss = []
all_dev_loss = []
all_dev_accuracy = []

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

    np.random.shuffle(train_data)

    for i in range(number_of_train_batches):
        train_batch = train_data[i*batch_size: (i+1)*batch_size]
        embed_index, hash_index, pos, chunk, label, length, sentence = prepare.prepare_ner(batch=train_batch, form='BIOES', gram='tri')

        pos = np.array([(np.concatenate([np_utils.to_categorical(p, pos_length), np.zeros((step_length-length[l], pos_length))])) for l,p in enumerate(pos)])
        # chunk = np.array([(np.concatenate([np_utils.to_categorical(c, chunk_length), np.zeros((step_length-length[l], chunk_length))])) for l,c in enumerate(chunk)])
        gazetteer, length_2 = prepare.prepare_gazetteer(batch=train_batch)
        gazetteer = np.array([(np.concatenate([a, np.zeros((step_length-length_2[l], gazetteer_length))])) for l,a in enumerate(gazetteer)])
        y = np.array([np_utils.to_categorical(each, output_length) for each in label])

        train_metrics = model.train_on_batch([embed_index, hash_index, pos, gazetteer], y)
        train_loss += train_metrics[0]
    all_train_loss.append(train_loss)

    correct_predict = 0
    all_predict = 0

    for j in range(number_of_dev_batches):
        dev_batch = dev_data[j*batch_size: (j+1)*batch_size]
        embed_index, hash_index, pos, chunk, label, length, sentence = prepare.prepare_ner(batch=dev_batch, form='BIOES', gram='tri')

        pos = np.array([(np.concatenate([np_utils.to_categorical(p, pos_length), np.zeros((step_length-length[l], pos_length))])) for l,p in enumerate(pos)])
        # chunk = np.array([(np.concatenate([np_utils.to_categorical(c, chunk_length), np.zeros((step_length-length[l], chunk_length))])) for l,c in enumerate(chunk)])
        gazetteer, length_2 = prepare.prepare_gazetteer(batch=dev_batch)
        gazetteer = np.array([(np.concatenate([a, np.zeros((step_length-length_2[l], gazetteer_length))])) for l,a in enumerate(gazetteer)])
        y = np.array([np_utils.to_categorical(each, output_length) for each in label])
        # for loss
        dev_metrics = model.test_on_batch([embed_index, hash_index, pos, gazetteer], y)
        dev_loss += dev_metrics[0]

        # for accuracy
        prob = model.predict_on_batch([embed_index, hash_index, pos, gazetteer])
        for i, l in enumerate(length):
            predict_label = np_utils.categorical_probas_to_classes(prob[i])
            correct_predict += np.sum(predict_label[:l]==label[i][:l])
        all_predict += np.sum(length)
    epcoh_accuracy = float(correct_predict)/all_predict
    all_dev_accuracy.append(epcoh_accuracy)

    all_dev_loss.append(dev_loss)

    if epcoh_accuracy>=best_accuracy:
        best_accuracy = epcoh_accuracy
        best_epoch = epoch

    end = datetime.now()

    model.save('%s/model_epoch_%d.h5'%(folder_path, epoch), overwrite=True)

    print('epoch %d end at %s'%(epoch, str(end)))
    print('epoch %d train loss: %f'%(epoch, train_loss))
    print('epoch %d dev loss: %f'%(epoch, dev_loss))
    print('epoch %d dev accuracy: %f'%(epoch, epcoh_accuracy))
    print('best epoch now: %d\n'%best_epoch)

    log.write('epoch %d end at %s\n'%(epoch, str(end)))
    log.write('epoch %d train loss: %f\n'%(epoch, train_loss))
    log.write('epoch %d dev loss: %f\n'%(epoch, dev_loss))
    log.write('epoch %d dev accuracy: %f\n'%(epoch, epcoh_accuracy))
    log.write('best epoch now: %d\n\n'%best_epoch)


end_time = datetime.now()
print('train end at %s\n'%str(end_time))
log.write('train end at %s\n\n'%str(end_time))

timedelta = end_time - start_time
print('train cost time: %s\n'%str(timedelta))
print('best epoch last: %d\n'%best_epoch)

log.write('train cost time: %s\n\n'%str(timedelta))
log.write('best epoch last: %d\n\n'%best_epoch)

plot.plot_loss(all_train_loss, all_dev_loss, folder_path=folder_path, title='%s'%model_name)
plot.plot_accuracy(all_dev_accuracy, folder_path=folder_path, title='%s'%model_name)
