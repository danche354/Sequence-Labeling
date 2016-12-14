'''
extract auto-encoder embedding index
'''

import json

filepath = 'conll2000-word.lst'

word_dict = {}

with open(filepath, 'r') as f:
    word_list = f.read().split()
    for i, word in enumerate(word_list):
        # 0 for masking
        word_dict[word] = i+1

with open('auto_encoder.json', 'w') as output:
    json.dump(word_dict, output)
