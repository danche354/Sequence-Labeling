'''
produce letter-3-gram representations
'''

# with open('../../dataset/chunk/train.txt') as tr:
#     train_sets = tr.read().split('\n')

# with open('../../dataset/chunk/test.txt') as te:
#     test_sets = te.read().split('\n')

# for word_pos_chunk in train_sets:
#     if word_pos_chunk:
#         word, tag, chunktag = word_pos_chunk.split()
#         word = '#'+word.lower()+'#'
#         word_length = len(word) - 2
#         for i in range(word_length):
#             if word[i:i+3] in l3g:
#                 l3g[word[i:i+3]] += 1
#             else:
#                 l3g[word[i:i+3]] = 1


letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g',\
            'h', 'i', 'j', 'k', 'l', 'm', 'n',\
            'o', 'p', 'q', 'r', 's', 't',\
            'u', 'v', 'w', 'x', 'y', 'z',\
            '0', '#']

l3g = {}

for pre in letter:
    for _ in letter:
        for suf in letter:
            l3g[pre+_+suf] = 1

sort_l3g = sorted(l3g.items(), key=lambda x:x[1], reverse=True)


_l3g = open('./l3g.txt', 'w')
_sorted_l3g = open('./sorted_l3g.txt', 'w')
for each in l3g:
    _l3g.write(each+'\n')
for each in sort_l3g:
    _sorted_l3g.write(str(each)+'\n')

