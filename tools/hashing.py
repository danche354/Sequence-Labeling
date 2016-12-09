'''
letter-3-gram hashing
convert word or sentence to letter-3-gram representation matrix
'''

import re

import numpy as np
from scipy.sparse import csc_matrix

import conf
from load_l3g import l3g_dict

feature_length = conf.feature_length

def pre_process(word_list, trigram=False):
    if trigram:
        # add start and end mark
        wrapper = ['#']
        wrapper.extend(word_list)
        wrapper.append('#')
        word_list = ['#'+word.strip()+'#' for word in wrapper]
    else:
        word_list = ['#'+word.strip()+'#' for word in word_list]
    return word_list

def word2index(word_list):
    word_list_length = len(word_list)
    sen_matrix = np.zeros((word_list_length, feature_length))
    for i, word in enumerate(word_list):
        length = len(word) - 2
        lword = word.lower()
        for j in range(length):
            sen_matrix[i, l3g_dict[lword[j: j+3]]] += 1
    return sen_matrix

def sen2matrix(word_list, trigram=False):
    if trigram:
        word_list = pre_process(word_list, trigram)
    else:
        word_list = pre_process(word_list)
    sen_matrix = word2index(word_list)
    sparse_sen_matrix = csc_matrix(sen_matrix)
    return sparse_sen_matrix