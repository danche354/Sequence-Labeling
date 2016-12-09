
import numpy as np

import hashing
import conf

step_length = conf.step_length
feature_length = conf.feature_length

NP_IOB = {'B-NP':0, 'I-NP':1, 'O':2}
ALL_IOB = {'B-ADVP':0, 'I-ADVP':1, 'B-ADJP':2, 'I-ADJP':3,
    'B-CONJP':4, 'I-CONJP':5, 'B-INTJ':6, 'I-INTJ':7,
    'B-LST':8, 'I-LST':9, 'B-NP':10, 'I-NP':11,
    'B-PP':12, 'I-PP':13, 'B-PRT':14, 'I-PRT':15,
    'B-SBAR':16, 'I-SBAR':17, 'B-UCP':18,
    'I-UCP':19, 'B-VP':20, 'I-VP':21, 'O':22}

POS = {'NN':0, 'IN':1, 'NNP':2, 'DT':3, 'NNS':4,
    'JJ':5, ',':6, '.':7, 'CD':8, 'VBD':9, 'RB':10, 'VB':11,
    'CC':12, 'TO':13, 'VBN':14, 'VBZ':15, 'PRP':16, 'VBG':17,
    'VBP':18, 'MD':19, 'PRP$':20, 'POS':21, '$':22, '``':23,
    "''":24, ':':25, 'WDT':26, 'JJR':27, 'WP':28, 'WRB':29,
    'NNPS':30, 'JJS':31, 'RBR':32, ')':33, '(':34, 'EX':35,
    'RBS':36, 'RP':37, 'PDT':38, '#':39, 'FW':40, 'WP$':41, 'UH':42, 'SYM':43}

def prepare_chunk_encoder(batch):
    X_hashing = hashing.sen2matrix(batch)
    return np.array(X_hashing)

def prepare_chunk(batch, chunk_type='ALL', step_length=step_length, feature_length=feature_length):
    if chunk_type=='NP':
        IOB = NP_IOB
    else:
        IOB = ALL_IOB

    X_hashing = []
    X_pos = []
    y = []
    sentence_length = []

    for sentence in batch:
        sen_matrix = hashing.sen2matrix(sentence[0])
        pos = [POS[each] for each in sentence[1]]
        label = [IBO[each] for each in sentence[2]]
        length = len(label)

        X_hashing.append(sen_matrix)
        X_pos.append(pos)
        y.append(label)
        # record the sentence length for calculate accuracy
        sentence_length.append(length)

    return np.array(X_hashing), np.array(X_pos), np.array(y), np.array(sentence_length)




