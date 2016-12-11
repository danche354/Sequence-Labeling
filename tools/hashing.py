'''
letter-3-gram hashing
convert word or sentence to letter-3-gram representation matrix
'''

import re

import numpy as np
from scipy.sparse import csc_matrix

import conf

feature_length = conf.feature_length
l3g_dict = conf.l3g_dict

letter = re.compile(r'[a-z]')

digit

def pre_process(word_list):
    word_list = ['#'+word.strip()+'#' for word in word_list]
    return word_list

def word2index(word_list):
    word_list_length = len(word_list)
    sen_matrix = np.zeros((word_list_length, feature_length))
    for i, word in enumerate(word_list):
        length = len(word) - 2
        # for letter preprocessing
        lword = ''
        for w in word.lower():
            if w.isalpha():
                lword += w
            elif w.isdigit():
                lword += '0'
            else:
                lword += '#'
        for j in range(length):
            sen_matrix[i, l3g_dict[lword[j: j+3]]] += 1
    return sen_matrix

def sen2matrix(word_list):
    word_list = pre_process(word_list)
    sen_matrix = word2index(word_list)
    sparse_sen_matrix = csc_matrix(sen_matrix)
    return sparse_sen_matrix