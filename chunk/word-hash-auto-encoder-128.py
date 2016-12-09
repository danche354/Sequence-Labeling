# from keras.models import Model
# from keras.layers import Input, Masking, Dense, LSTM
# from keras.layers import Dropout, merge, Bidirectional
# from keras.layers.embeddings import Embedding
# from keras.utils import np_utils

import numpy as np

import sys
import math
import os

# add path
sys.path.append('../')

# import load_data
# import senna_letter_prepare_data
# import word_prepare_data
from tools import load_data
from tools import conf

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
(X_train, y_train, X_dev, y_dev, X_test, y_test) = load_data.load_chunk(amount=0, split_rate=split_rate)

train_samples = len(y_train) + len(y_dev)
dev_samples = len(y_test)
print('train shape:', train_samples)
print('test shape:', dev_samples)

word_train =[]
word_dev = []
# all train sample, combine train and dev
[word_train.extend(list(each[0])) for each in X_train]
[word_train.extend(list(each[0])) for each in X_dev]
[word_test.extend(list(each[0])) for each in X_test]

word_train_samples=len(word_train)
word_dev_samples=len(word_test)
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

accuracy = []
all_error = 1000
best_epoch = 0

log = open('./log/%s/model_log.txt'%model_name, 'w')

for epoch in range(word_nb_epoch):
    print('#'*50)
    print('Train epoch: ', epoch)
    print('#'*50)

    log.write('#'*50+'\n')
    log.write('Train epoch: %d'%epoch+'\n')
    log.write('#'*50+'\n')

    epoch_loss = []
    loss = 0

    np.random.shuffle(word_train)

    for i in range(number_of_train_batches):
        train_batch = word_train[i*batch_size: (i+1)*batch_size]
        X = word_prepare_data.prepare(batch=train_batch)
        X = X.toarray()

        # P_batch = np.array([(np.concatenate([np_utils.to_categorical(p, 44), np.zeros((step_length-length[l], 44))])) for l,p in enumerate(P)])
        # Y_batch = np.array([np_utils.to_categorical(y, 3) for y in Y])
        metrics = model.train_on_batch(X, X)
        loss += metrics[0]
        print('='*5+'train batch %s'%str(i)+'='*5+'loss: ', metrics[0], '     accuracy: ', metrics[1])
        log.write('='*5+'train batch %s'%str(i)+'='*5+'loss: %f'%metrics[0]+'     accuracy: %f'%metrics[1]+'\n')

    plt.plot(loss)
    plt.title('epoch_%d_loss'%epoch)
    plt.savefig('./fig/%s/model_2_epoch_%d_loss'%(model_name, epoch))
    plt.close()
    
    model_2.save('model/%s/model_2_epoch_%s.h5'%(model_name, str(epoch)), overwrite=True)
    model_2_hidden.save('model/%s/model_2_hidden_epoch_%s.h5'%(model_name, str(epoch)), overwrite=True)

    print('\n')
    print('#'*50)
    print('Test epoch: ', epoch)
    print('#'*50)

    log.write('#'*50+'\n')
    log.write('Test epoch: %d'%epoch+'\n')
    log.write('#'*50+'\n')
    log.write('\n')

    error = 0
    all_error = 0

    for j in range(number_of_word_test_batches):
        sample_list_test = word_test[j*word_batch_size: (j+1)*word_batch_size]
        X_2 = word_prepare_data.prepare(sample_list=sample_list_test)
        X_2 = X_2.toarray()
        # P_batch = np.array([(np.concatenate([np_utils.to_categorical(p, 44), np.zeros((step_length-length[i], 44))])) for i,p in enumerate(P)])
        prob = model_2.predict_on_batch(X_2)
        # for i, x in enumerate(X_2):
        error = ((X_2 - prob) ** 2).mean()
        all_error += error
        print('='*5+'test batch %s'%str(j)+'='*5+'accumulation accuracy: ', all_error)
        log.write('='*5+'test batch %s'%str(j)+'='*5+'accumulation accuracy: %f'%all_error+'\n')

    if all_error<best_accuracy_2:
        best_accuracy_2 = all_error
        best_epoch_2 = epoch
        print('*'*50)
        print('model_2_hidden best epoch', epoch)
        print('*'*50)
        model_2_best = model_2_hidden
    accuracy.append(all_error)
    print('='*10+'test error'+'='*10+': ', all_error)
    print('\n\n'+'+'*100+'\n\n')

    log.write('='*10+'test error'+'='*10+': %f'%all_error+'\n')
    log.write('\n\n'+'+'*100+'\n\n')

print('model 2 best epoch:', best_epoch_2)

plt.plot(accuracy)
plt.title('epoch test accuracy')
plt.savefig('./fig/%s/model_2_epoch_test_accuracy'%model_name)

print("model_2 train over!")