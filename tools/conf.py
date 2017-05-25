'''
config file
'''
import json

with open('../preprocessing/senna/senna.json') as j:
    senna_dict = json.load(j)

with open('../preprocessing/chunk-auto-encoder/auto_encoder.json') as l:
    chunk_hash_dict = json.load(l)

with open('../preprocessing/chunk-auto-encoder-2/auto_encoder.json') as m:
    chunk_hash_dict_2 = json.load(m)

chunk_l3g_dict = {}

with open('../preprocessing/chunk-l3g/l3g.txt') as f:
    chunk_l3g_list = f.read().strip().split('\n')
    for i, each in enumerate(chunk_l3g_list):
        chunk_l3g_dict[each] = i

chunk_l2g_dict = {}

with open('../preprocessing/chunk-l2g/l2g.txt') as g:
    chunk_l2g_list = g.read().strip().split('\n')
    for i, each in enumerate(chunk_l2g_list):
        chunk_l2g_dict[each] = i

with open('../preprocessing/ner-auto-encoder/auto_encoder.json') as l:
    ner_hash_dict = json.load(l)

with open('../preprocessing/ner-auto-encoder-2/auto_encoder.json') as m:
    ner_hash_dict_2 = json.load(m)

ner_l3g_dict = {}

with open('../preprocessing/ner-l3g/l3g.txt') as f:
    ner_l3g_list = f.read().strip().split('\n')
    for i, each in enumerate(ner_l3g_list):
        ner_l3g_dict[each] = i

ner_l2g_dict = {}

with open('../preprocessing/ner-l2g/l2g.txt') as f:
    ner_l2g_list = f.read().strip().split('\n')
    for i, each in enumerate(ner_l2g_list):
        ner_l2g_dict[each] = i



def gazetteer(data):
    word_dict = {}
    with open(data, 'r') as f:
        word_list = f.read().strip().split('\n')
        # word_list = []
        for i, word in enumerate(word_list):
            # 0 for masking
            word_dict[word] = i
    return word_dict

LOC = gazetteer('../preprocessing/senna/ner.loc.lst')        
PER = gazetteer('../preprocessing/senna/ner.per.lst')        
ORG = gazetteer('../preprocessing/senna/ner.org.lst')        
MISC = gazetteer('../preprocessing/senna/ner.misc.lst')        
gazetteer_length = 4
BIOES_gazetteer_length = 16

LOC_conll = {}
PER_conll = {}
ORG_conll = {}
MISC_conll = {}

with open('../preprocessing/gazetteer/eng.list', 'r') as f:
    word_list = f.read().strip().split('\n')
    # word_list = []
    for i, word in enumerate(word_list):
        word_split = word.split(" ", 1)
        key = word_split[0].strip()
        value = word_split[1].strip()
        if key=='LOC':
            LOC_conll[value] = i
        elif key=='PER':
            PER_conll[value] = i
        elif key=='ORG':
            ORG_conll[value] = i
        elif key=='MISC':
            MISC_conll[value] = i
        else:
            raise Exception('NAME ERROR!')

chunk_step_length = 80
chunk_feature_length = 8616

chunk_feature_length_2 = 1072

ner_step_length = 126
ner_feature_length = 11313
ner_feature_length_2 = 1233

senna_length = 50
senna_vocab = 130000

chunk_hash_length = 128
chunk_hash_vocab = 19460

ner_hash_length = 128
ner_hash_vocab = 26870

chunk_pos_length = 44

ner_pos_length = 46


chunk_NP_length = 3
chunk_ALL_length = 23

additional_length = 9

chunk_split_rate = 0.9

ner_chunk_length = 18
ner_IOB_length = 9
ner_BIOES_length = 17

word_batch_size = 200
batch_size = 20

word_nb_epoch = 30

nb_epoch = 40

chunk_NP_IOB_encode = {'B-NP':0, 'I-NP':1, 'O':2}
chunk_NP_IOB_decode = {0: 'B-NP', 1: 'I-NP', 2: 'O'}

ner_IOB_encode = {'B-LOC':0, 'I-LOC':1, 'B-MISC':2, 'I-MISC':3,
            'B-ORG':4, 'I-ORG':5, 'B-PER':6, 'I-PER':7, 'O':8}
ner_IOB_decode = {0:'B-LOC', 1:'I-LOC', 2:'B-MISC', 3:'I-MISC',
            4:'B-ORG', 5:'I-ORG', 6:'B-PER', 7:'I-PER', 8:'O'}

ner_BIOES_encode = {'B-LOC':0, 'I-LOC':1, 'E-LOC':2, 'S-LOC':3,
                    'B-MISC':4, 'I-MISC':5, 'E-MISC':6, 'S-MISC':7,
                    'B-ORG':8, 'I-ORG':9, 'E-ORG':10, 'S-ORG':11,
                    'B-PER':12, 'I-PER':13, 'E-PER':14, 'S-PER':15, 'O':16}
ner_BIOES_decode = {0:'B-LOC', 1:'I-LOC', 2:'E-LOC', 3:'S-LOC',
                    4:'B-MISC', 5:'I-MISC', 6:'E-MISC', 7:'S-MISC',
                    8:'B-ORG', 9:'I-ORG', 10:'E-ORG', 11:'S-ORG',
                    12:'B-PER', 13:'I-PER', 14:'E-PER', 15:'S-PER', 16:'O'}


chunk_ALL_IOB_encode = {'B-ADVP':0, 'I-ADVP':1, 'B-ADJP':2, 'I-ADJP':3,
    'B-CONJP':4, 'I-CONJP':5, 'B-INTJ':6, 'I-INTJ':7,
    'B-LST':8, 'I-LST':9, 'B-NP':10, 'I-NP':11,
    'B-PP':12, 'I-PP':13, 'B-PRT':14, 'I-PRT':15,
    'B-SBAR':16, 'I-SBAR':17, 'B-UCP':18,
    'I-UCP':19, 'B-VP':20, 'I-VP':21, 'O':22}

chunk_ALL_IOB_decode = {0:'B-ADVP', 1:'I-ADVP', 2:'B-ADJP', 3:'I-ADJP',
    4:'B-CONJP', 5:'I-CONJP', 6:'B-INTJ', 7:'I-INTJ',
    8:'B-LST', 9:'I-LST', 10:'B-NP', 11:'I-NP',
    12:'B-PP', 13:'I-PP', 14:'B-PRT', 15:'I-PRT',
    16:'B-SBAR', 17:'I-SBAR', 18:'B-UCP',
    19:'I-UCP', 20:'B-VP', 21:'I-VP', 22:'O'}

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
    'RBS':36, 'RP':37, 'PDT':38, 'LS':39, 'FW':40, 'WP$':41, 'UH':42, 'SYM':43,
    '"':44, 'NN|SYM':45}
