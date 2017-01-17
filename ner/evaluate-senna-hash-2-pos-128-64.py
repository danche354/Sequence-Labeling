'''
evaluate result
'''

from keras.models import load_model
from keras.utils import np_utils

import numpy as np

import os
import sys

# add path
sys.path.append('../')
sys.path.append('../tools')


from tools import conf
from tools import load_data
from tools import prepare

# input sentence dimensions
step_length = conf.ner_step_length
pos_length = conf.ner_pos_length
chunk_length = conf.ner_chunk_length

IOB = conf.ner_IOB_decode

test_data = load_data.load_ner(dataset='eng.testb')

best_epoch = sys.argv[1]

model_name = os.path.basename(__file__)[9:-3]
folder_path = './model/%s'%model_name

model_path = '%s/model_epoch_%s.h5'%(folder_path, best_epoch)
result = open('%s/predict.txt'%folder_path, 'w')


print('loading model...')
model = load_model(model_path)
print('loading model finished.')

for each in test_data:
    embed_index, hash_index, pos, chunk, label, length, sentence = prepare.prepare_ner(batch=[each], gram='bi')
    pos = np.array([(np.concatenate([np_utils.to_categorical(p, pos_length), np.zeros((step_length-length[l], pos_length))])) for l,p in enumerate(pos)])
    prob = model.predict_on_batch([embed_index, hash_index, pos])

    for i, l in enumerate(length):
        predict_label = np_utils.categorical_probas_to_classes(prob[i])
        chunktags = [IOB[j] for j in predict_label][:l]

    word_pos_chunk = list(zip(*each))

    for ind, chunktag in enumerate(chunktags):
        result.write(' '.join(word_pos_chunk[ind])+' '+chunktag+'\n')
    result.write('\n')

result.close()
print('epoch %s predict over !'%best_epoch)

os.system('../tools/conlleval < %s/predict.txt'%folder_path)

