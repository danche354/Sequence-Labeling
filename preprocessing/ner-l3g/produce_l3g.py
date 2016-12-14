'''
produce letter-3-gram representations
'''

with open('../../dataset/ner/eng.train') as tr:
    train_sets = tr.read().split('\n')

with open('../../dataset/ner/eng.testa') as ta:
    test_setsa = ta.read().split('\n')

with open('../../dataset/ner/eng.testb') as tb:
    test_setsb = tb.read().split('\n')

l3g = {}

for data in [train_sets, test_setsa, test_setsb]:
    for word_pos_chunk_ner in data:
        if word_pos_chunk_ner:
            word, tag, chunktag, ner = word_pos_chunk_ner.split()
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

