'''
config file
'''
import json

with open('../preprocessing/senna/senna.json') as j:
    senna_dict = json.load(j)

with open('../preprocessing/auto-encoder/auto_encoder.json') as l:
    auto_encoder_dict = json.load(l)

l3g_dict = {}

with open('../preprocessing//l3g/l3g.txt') as f:
    l3g_list = f.read().strip().split('\n')
    for i, each in enumerate(l3g_list):
        l3g_dict[each] = i

step_length = 80
feature_length = 8146
senna_length = 50
senna_vocab = 130000

auto_length = 128
auto_vocab = 19460

pos_length = 44

NP_length = 3
ALL_length = 23

split_rate = 0.9

word_batch_size = 400
batch_size = 20

nb_epoch = 20

NP_IOB_encode = {'B-NP':0, 'I-NP':1, 'O':2}
NP_IOB_decode = {0: 'B-NP', 1: 'I-NP', 2: 'O'}

ALL_IOB_encode = {'B-ADVP':0, 'I-ADVP':1, 'B-ADJP':2, 'I-ADJP':3,
    'B-CONJP':4, 'I-CONJP':5, 'B-INTJ':6, 'I-INTJ':7,
    'B-LST':8, 'I-LST':9, 'B-NP':10, 'I-NP':11,
    'B-PP':12, 'I-PP':13, 'B-PRT':14, 'I-PRT':15,
    'B-SBAR':16, 'I-SBAR':17, 'B-UCP':18,
    'I-UCP':19, 'B-VP':20, 'I-VP':21, 'O':22}

POS_encode = {'NN':0, 'IN':1, 'NNP':2, 'DT':3, 'NNS':4,
    'JJ':5, ',':6, '.':7, 'CD':8, 'VBD':9, 'RB':10, 'VB':11,
    'CC':12, 'TO':13, 'VBN':14, 'VBZ':15, 'PRP':16, 'VBG':17,
    'VBP':18, 'MD':19, 'PRP$':20, 'POS':21, '$':22, '``':23,
    "''":24, ':':25, 'WDT':26, 'JJR':27, 'WP':28, 'WRB':29,
    'NNPS':30, 'JJS':31, 'RBR':32, ')':33, '(':34, 'EX':35,
    'RBS':36, 'RP':37, 'PDT':38, '#':39, 'FW':40, 'WP$':41, 'UH':42, 'SYM':43}
