'''
produce letter-2-gram representations
'''

with open('../../dataset/chunk/train.txt') as tr:
    train_sets = tr.read().split('\n')

with open('../../dataset/chunk/test.txt') as te:
    test_sets = te.read().split('\n')

l2g = {}

for data in [train_sets, test_sets]:
    for word_pos_chunk in data:
        if word_pos_chunk:
            word, tag, chunktag = word_pos_chunk.split()
            word = '#'+word.lower()+'#'
            word_lenght = len(word) - 1
            for i in range(word_lenght):
                if word[i:i+2] in l2g:
                    l2g[word[i:i+2]] += 1
                else:
                    l2g[word[i:i+2]] = 1

sort_l2g = sorted(l2g.items(), key=lambda x:x[1], reverse=True)

a = open('./l2g.txt', 'w')
b = open('./sorted_l2g.txt', 'w')
for each in l2g:
    a.write(each+'\n')
for each in sort_l2g:
    b.write(str(each)+'\n')

