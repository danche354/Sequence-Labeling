
import numpy as np

import hashing
import conf

word_dict = conf.word_dict

step_length = conf.step_length
feature_length = conf.feature_length
emb_vocab = conf.senna_vocab

ALL_IOB = conf.ALL_IOB_encode
NP_IOB = conf.NP_IOB_encode
POS = conf.POS_encode

def prepare_chunk_encoder(batch):
    X_hashing = hashing.sen2matrix(batch)
    return X_hashing

def prepare_chunk(batch, chunk_type='NP', step_length=step_length, feature_length=feature_length):
    if chunk_type=='ALL':
        IOB = ALL_IOB
    else:
        IOB = NP_IOB

    embedding_index = []
    X_hashing = []
    X_pos = []
    y = []
    sentence_length = []
    sentences = []

    for sentence in batch:
        sentence_index = [word_dict.get(each.strip().lower(), emb_vocab+1) for each in sentence[0]]
        sen_matrix = hashing.sen2matrix(sentence[0])
        sentences.append(sentence[0])
        pos = [POS[each] for each in sentence[1]]
        label = [IOB[each] for each in sentence[2]]
        length = len(label)

        label.extend([0]*(step_length-length))
        sentence_index.extend([0]*(step_length-length))

        embedding_index.append(sentence_index)
        X_hashing.append(sen_matrix)
        X_pos.append(pos)
        y.append(label)
        # record the sentence length for calculate accuracy
        sentence_length.append(length)

    return np.array(embedding_index), np.array(X_hashing), np.array(X_pos), np.array(y), np.array(sentence_length), sentences




