
import numpy as np

import hashing
import conf

embedding_dict = conf.senna_dict
emb_vocab = conf.senna_vocab

auto_encoder_dict = conf.auto_encoder_dict
auto_vocab = conf.auto_vocab

step_length = conf.step_length
feature_length = conf.feature_length

ALL_IOB = conf.ALL_IOB_encode
NP_IOB = conf.NP_IOB_encode
POS = conf.POS_encode

def prepare_chunk_encoder(batch):
    hashing = hashing.sen2matrix(batch)
    return hashing

def prepare_chunk(batch, trigram=False, chunk_type='NP', step_length=step_length, feature_length=feature_length):
    if chunk_type=='ALL':
        IOB = ALL_IOB
    else:
        IOB = NP_IOB

    embedding_index = []
    auto_encoder_index = []
    pos = []
    label = []
    sentence_length = []
    sentences = []

    for sentence in batch:
        # sentence and sentence pos
        sequence = list(sentence[0])
        sequence_pos = list(sentence[1])
        if trigram:
            # add start and end mark
            sequence.insert(0, '#')
            sequence.append('#')
            sequence_pos.insert(0, '#')
            sequence_pos.append('#')

        _embedding_index = [embedding_dict.get(each.strip().lower(), emb_vocab+1) for each in sequence]
        _auto_encoder_index = [auto_encoder_dict.get(each.strip().lower(), auto_vocab+1) for each in sequence]
        _sentence.append(sentence[0])
        _pos = [POS[each] for each in sequence_pos]
        _label = [IOB[each] for each in sentence[2]]
        length = len(label)

        _label.extend([0]*(step_length-length))
        _embedding_index.extend([0]*(step_length-length))
        _auto_encoder_index.extend([0]*(step_length-length))

        embedding_index.append(_embedding_index)
        auto_encoder_index.append(_auto_encoder_index)
        pos.append(_pos)
        label.append(_label)
        # record the sentence length for calculate accuracy
        sentence_length.append(length)

    return np.array(embedding_index), np.array(auto_encoder_index), np.array(pos), np.array(label), np.array(sentence_length), sentences




