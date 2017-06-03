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
gazetteer_length = conf.gazetteer_length
feature_length = conf.ner_feature_length_2

IOB = conf.ner_BIOES_decode

data = sys.argv[1]

best_epoch = sys.argv[2]

if data=="dev":
    test_data = load_data.load_ner(dataset='eng.testa', form='BIOES')
elif data == "test":
    test_data = load_data.load_ner(dataset='eng.testb', form='BIOES')
tokens = [len(x[0]) for x in test_data]
print(sum(tokens))
print('%s shape:'%data, len(test_data))

model_name = os.path.basename(__file__)[9:-3]
folder_path = './model/%s'%model_name

model_path = '%s/model_epoch_%s.h5'%(folder_path, best_epoch)
result = open('%s/predict.txt'%folder_path, 'w')

def convert(chunktags):
    # convert BIOES to BIO
    for p, q in enumerate(chunktags):
        if q.startswith("E-"):
            chunktags[p] = "I-" + q[2:]
        elif q.startswith("S-"):
            if p==0:
                chunktags[p] = "I-" + q[2:]
            elif q[2:]==chunktags[p-1][2:]:
                chunktags[p] = "B-" + q[2:]
            elif q[2:]!=chunktags[p-1][2:]:
                chunktags[p] = "I-" + q[2:]
        elif q.startswith("B-"):
            if p==0:
                chunktags[p] = "I-" + q[2:]
            else:
                if q[2:]!=chunktags[p-1][2:]:
                    chunktags[p] = "I-" + q[2:]
    return chunktags

print('loading model...')
model = load_model(model_path)
print('loading model finished.')

for each in test_data:
    embed_index, hash_representation, pos, chunk, label, length, sentence = prepare.prepare_ner_raw(batch=[each], gram='bi', form='BIOES')
    
    hash_representation = [each_h.toarray() for each_h in hash_representation]
    hash_representation = np.array([np.concatenate([h, np.zeros((step_length-length[l], feature_length))]) for l, h in enumerate(hash_representation)])

    pos = np.array([(np.concatenate([np_utils.to_categorical(p, pos_length), np.zeros((step_length-length[l], pos_length))])) for l,p in enumerate(pos)])
#    chunk = np.array([(np.concatenate([np_utils.to_categorical(c, chunk_length), np.zeros((step_length-length[l], chunk_length))])) for l,c in enumerate(chunk)])
    gazetteer, length_2 = prepare.prepare_gazetteer(batch=[each])
    gazetteer = np.array([(np.concatenate([a, np.zeros((step_length-length_2[l], gazetteer_length))])) for l,a in enumerate(gazetteer)])
    prob = model.predict_on_batch([embed_index, hash_representation, pos, gazetteer])

    for i, l in enumerate(length):
        predict_label = np_utils.categorical_probas_to_classes(prob[i])
        chunktags = [IOB[j] for j in predict_label][:l]

    word_pos_chunk = list(zip(*each))
    
    # convert
    word_pos_chunk = list(zip(*word_pos_chunk))
    word_pos_chunk = [list(x) for x in word_pos_chunk]
    # if data == "test":
        # word_pos_chunk[3] = convert(word_pos_chunk[3])
    word_pos_chunk = list(zip(*word_pos_chunk))

    #convert
    # if data == "test":
        # chunktags = convert(chunktags) 

    for ind, chunktag in enumerate(chunktags):
        result.write(' '.join(word_pos_chunk[ind])+' '+chunktag+'\n')
    result.write('\n')

result.close()
print('epoch %s predict over !'%best_epoch)

os.system('../tools/conlleval < %s/predict.txt'%folder_path)

