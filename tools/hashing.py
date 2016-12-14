'''
letter-3-gram hashing
convert word or sentence to letter-3-gram representation matrix
'''

import numpy as np
from scipy.sparse import csc_matrix

import conf

chunk_feature_length = conf.chunk_feature_length
chunk_l3g_dict = conf.chunk_l3g_dict

ner_feature_length = conf.ner_feature_length
ner_l3g_dict = conf.ner_l3g_dict

def pre_process(word_list):
    word_list = ['#'+word.strip()+'#' for word in word_list]
    return word_list

def word2index(word_list, task):
    
    if task=='chunk':
        feature_length = chunk_feature_length
        l3g_dict = chunk_l3g_dict
    elif task='ner':
        feature_length = ner_feature_length
        l3g_dict = ner_l3g_dict

    word_list_length = len(word_list)
    sen_matrix = np.zeros((word_list_length, feature_length))
    for i, word in enumerate(word_list):
        length = len(word) - 2
        # for letter preprocessing
        lword = word.lower()
        for j in range(length):
            sen_matrix[i, l3g_dict[lword[j: j+3]]] += 1
    return sen_matrix

def sen2matrix(word_list, task):
    word_list = pre_process(word_list)
    sen_matrix = word2index(word_list, task)
    sparse_sen_matrix = csc_matrix(sen_matrix)
    return sparse_sen_matrix