'''
load datasets
'''

import numpy as np

np.random.seed(1337)

def load_chunk(amount=0, split_rate=0.9, chunk_type='ALL'):

    train_set = open('../dataset/chunk/train.txt', 'r')
    train_set = train_set.read().strip().split('\n\n')

    if amount!=0:
        train_set = train_set[:amount]

    np.random.shuffle(train_set)

    train_set = list(map(str2tuple, train_set))
    if chunk_type=='NP':
        for sentence in train_set:
            chunk_tags = list(sentence[2])
            for ind, chunk in enumerate(chunk_tags):
                if chunk!='B-NP' and chunk!='I-NP':
                    chunk_tags[ind] = 'O'
            sentence[2] = tuple(chunk_tags)


    length = len(train_set)
    margin = int(length*split_rate)

    train_data = train_set[:margin]
    dev_data = train_set[margin:]


    test_set = open('../dataset/chunk/test.txt', 'r')
    test_set = test_set.read().strip().split('\n\n')

    test_set = list(map(str2tuple, test_set))
    if chunk_type=='NP':
        for sentence in test_set:
            chunk_tags = list(sentence[2])
            for ind, chunk in enumerate(chunk_tags):
                if chunk!='B-NP' and chunk!='I-NP':
                    chunk_tags[ind] = 'O'
            sentence[2] = tuple(chunk_tags)

    test_data = test_set

    return train_data, dev_data, test_data

# sentence example:
# ['He PR B-NP', 'is VB I-VP']
def str2tuple(sentence):
    sentence_split = [each.split() for each in sentence.split('\n')]
    return list(zip(*sentence_split))

if __name__ == '__main__':
    # load_chunk()
    load_chunk(chunk_type='ALL')