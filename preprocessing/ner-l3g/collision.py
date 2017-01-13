'''
produce letter-3-gram representations
'''

with open('../../dataset/ner/eng.train') as tr:
    train_sets = tr.read().split('\n')

with open('../../dataset/ner/eng.testa') as ta:
    test_setsa = ta.read().split('\n')

with open('../../dataset/ner/eng.testb') as tb:
    test_setsb = tb.read().split('\n')

word_dict = {}

word_l3g = {}

for data in [train_sets, test_setsa, test_setsb]:
    for word_pos_chunk_ner in data:
        if word_pos_chunk_ner:
            l3g_temp = {}
            word, tag, chunktag, ner = word_pos_chunk_ner.split()
            word = '#'+word.lower()+'#'
            if word not in word_dict:
                word_dict[word] = 1
                word_lenght = len(word) - 1
                for i in range(word_lenght):
                    if word[i:i+2] in l3g_temp:
                        l3g_temp[word[i:i+2]] += 1
                    else:
                        l3g_temp[word[i:i+2]] = 1
                sort_l3g_temp = sorted(l3g_temp.items(), key=lambda x:x[0], reverse=True)

                tuple_sort_l3g_dict = tuple(sort_l3g_temp)
                if tuple_sort_l3g_dict in word_l3g:
                    word_l3g[tuple_sort_l3g_dict].append(word)
                else:
                    word_l3g[tuple_sort_l3g_dict] = []
                    word_l3g[tuple_sort_l3g_dict].append(word)

for k, v in word_l3g.items():
    if len(v)>1:
        print(k, v)

print(len(word_l3g))
