'''
produce letter-3-gram representations
'''

with open('../../dataset/chunk/train.txt') as tr:
    train_sets = tr.read().split('\n')

with open('../../dataset/chunk/test.txt') as te:
    test_sets = te.read().split('\n')

l3g = {}

for data in [train_sets, test_sets]:
    for word_pos_chunk in data:
        if word_pos_chunk:
            word, tag, chunktag = word_pos_chunk.split()
            word = '#'+word.lower()+'#'
            word_lenght = len(word) - 2
            for i in range(word_lenght):
                if word[i:i+3] in l3g:
                    l3g[word[i:i+3]] += 1
                else:
                    l3g[word[i:i+3]] = 1

sort_l3g = sorted(l3g.items(), key=lambda x:x[1], reverse=True)

a = open('./l3g.txt', 'w')
b = open('./sorted_l3g.txt', 'w')
for each in l3g:
    a.write(each+'\n')
for each in sort_l3g:
    b.write(str(each)+'\n')

