'''
config file
'''
import json

with open('../preprocessing/senna/senna.json') as j:
    senna_dict = json.load(j)

with open('../preprocessing/chunk-auto-encoder/auto_encoder.json') as l:
    chunk_hash_dict = json.load(l)

chunk_l3g_dict = {}

with open('../preprocessing/chunk-l3g/l3g.txt') as f:
    chunk_l3g_list = f.read().strip().split('\n')
    for i, each in enumerate(chunk_l3g_list):
        chunk_l3g_list[each] = i

with open('../preprocessing/ner-auto-encoder/auto_encoder.json') as l:
    ner_hash_dict = json.load(l)

ner_l3g_dict = {}

with open('../preprocessing/ner-l3g/l3g.txt') as f:
    ner_l3g_list = f.read().strip().split('\n')
    for i, each in enumerate(ner_l3g_list):
        ner_l3g_list[each] = i


chunk_step_length = 80
chunk_feature_length = 8616

ner_step_length = 80
ner_feature_length = 11313

senna_length = 50
senna_vocab = 130000

chunk_hash_length = 128
chunk_hash_vocab = 19460

ner_hash_length = 128
ner_hash_vocab = 19460

chunk_pos_length = 44

ner_pos_length = 44


chunk_NP_length = 3
chunk_ALL_length = 23

ner_chunk_length = 9

chunk_additional_length = 9

chunk_NP_length = 3
chunk_ALL_length = 23

chunk_split_rate = 0.9

word_batch_size = 200
batch_size = 20

nb_epoch = 30

chunk_NP_IOB_encode = {'B-NP':0, 'I-NP':1, 'O':2}
chunk_NP_IOB_decode = {0: 'B-NP', 1: 'I-NP', 2: 'O'}

ner_IOB_encode = {'B-LOC':0, 'I-LOC':1, 'B-MISC':2, 'I-MISC':3,
            'B-ORG':4, 'I-ORG':5, 'B-PER':6, 'I-PER':7, 'O':8}
ner_IOB_decode = {0:'B-LOC', 1:'I-LOC', 2:'B-MISC', 3:'I-MISC',
            4:'B-ORG', 5:'I-ORG', 6:'B-PER', 7:'I-PER', 8:'O'}

['B-LOC', 'B-ORG', 'B-MISC', 'I-LOC', 'I-MISC', 'O', 'I-ORG', 'I-PER']

chunk_ALL_IOB_encode = {'B-ADVP':0, 'I-ADVP':1, 'B-ADJP':2, 'I-ADJP':3,
    'B-CONJP':4, 'I-CONJP':5, 'B-INTJ':6, 'I-INTJ':7,
    'B-LST':8, 'I-LST':9, 'B-NP':10, 'I-NP':11,
    'B-PP':12, 'I-PP':13, 'B-PRT':14, 'I-PRT':15,
    'B-SBAR':16, 'I-SBAR':17, 'B-UCP':18,
    'I-UCP':19, 'B-VP':20, 'I-VP':21, 'O':22}

chunk_POS_encode = {'NN':0, 'IN':1, 'NNP':2, 'DT':3, 'NNS':4,
    'JJ':5, ',':6, '.':7, 'CD':8, 'VBD':9, 'RB':10, 'VB':11,
    'CC':12, 'TO':13, 'VBN':14, 'VBZ':15, 'PRP':16, 'VBG':17,
    'VBP':18, 'MD':19, 'PRP$':20, 'POS':21, '$':22, '``':23,
    "''":24, ':':25, 'WDT':26, 'JJR':27, 'WP':28, 'WRB':29,
    'NNPS':30, 'JJS':31, 'RBR':32, ')':33, '(':34, 'EX':35,
    'RBS':36, 'RP':37, 'PDT':38, '#':39, 'FW':40, 'WP$':41, 'UH':42, 'SYM':43}

ner_chunk_encode = {'B-ADVP':0, 'I-ADVP':1, 'B-ADJP':2, 'I-ADJP':3,
    'I-CONJP':4, 'I-INTJ':5, 'I-LST':6, 'B-NP':7, 'I-NP':8,
    'B-PP':9, 'I-PP':10, 'I-PRT':11, 'B-SBAR':12, 'I-SBAR':13,
    'B-VP':14, 'I-VP':15, 'O':16,'-X-':17}

ner_POS_encode = {'NN':0, 'IN':1, 'NNP':2, 'DT':3, 'NNS':4,
    'JJ':5, ',':6, '.':7, 'CD':8, 'VBD':9, 'RB':10, 'VB':11,
    'CC':12, 'TO':13, 'VBN':14, 'VBZ':15, 'PRP':16, 'VBG':17,
    'VBP':18, 'MD':19, 'PRP$':20, 'POS':21, '$':22, '-X-':23,
    "''":24, ':':25, 'WDT':26, 'JJR':27, 'WP':28, 'WRB':29,
    'NNPS':30, 'JJS':31, 'RBR':32, ')':33, '(':34, 'EX':35,
    'RBS':36, 'RP':37, 'PDT':38, '#':39, 'FW':40, 'WP$':41, 'UH':42, 'SYM':43,
    '"':44, 'NN|SYM':45}