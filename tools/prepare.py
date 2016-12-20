
import numpy as np

import re

import hashing
import conf

embedding_dict = conf.senna_dict
emb_vocab = conf.senna_vocab

chunk_hash_dict = conf.chunk_hash_dict
chunk_hash_vocab = conf.chunk_hash_vocab

ner_hash_dict = conf.ner_hash_dict
ner_hash_vocab = conf.ner_hash_vocab

chunk_step_length = conf.chunk_step_length
chunk_feature_length = conf.chunk_feature_length
chunk_additional_length = conf.chunk_additional_length


ner_step_length = conf.ner_step_length
ner_feature_length = conf.ner_feature_length

chunk_ALL_IOB = conf.chunk_ALL_IOB_encode
chunk_NP_IOB = conf.chunk_NP_IOB_encode
chunk_POS = conf.chunk_POS_encode


ner_chunk = conf.ner_chunk_encode
ner_POS = conf.ner_POS_encode
ner_IOB = conf.ner_IOB_encode


def prepare_auto_encoder(batch, task):
    word_hashing = hashing.sen2matrix(batch, task)
    return word_hashing

def prepare_chunk(batch, trigram=False, chunk_type='NP', step_length=chunk_step_length, feature_length=chunk_feature_length):
    if chunk_type=='ALL':
        IOB = chunk_ALL_IOB
    else:
        IOB = chunk_NP_IOB

    embedding_index = []
    hash_index = []
    pos = []
    label = []
    sentence_length = []
    sentences = []

    for sentence in batch:
        # sentence and sentence pos
        sequence = list(sentence[0])
        sequence_pos = list(sentence[1])
        # for trigram
        if trigram:
            # add start and end mark
            sequence.insert(0, '#')
            sequence.append('#')
            sequence_pos.insert(0, '#')
            sequence_pos.append('#')

        _embedding_index = [embedding_dict.get(each.strip().lower(), emb_vocab+1) for each in sequence]
        _hash_index = [chunk_hash_dict.get(each.strip().lower(), chunk_hash_vocab+1) for each in sequence]
        sentences.append(sentence[0])
        _pos = [chunk_POS[each] for each in sequence_pos]
        _label = [IOB[each] for each in sentence[2]]
        length = len(_label)

        _label.extend([0]*(chunk_step_length-length))
        _embedding_index.extend([0]*(chunk_step_length-length))
        _hash_index.extend([0]*(chunk_step_length-length))

        embedding_index.append(_embedding_index)
        hash_index.append(_hash_index)
        pos.append(_pos)
        label.append(_label)
        # record the sentence length for calculate accuracy
        sentence_length.append(length)

    return np.array(embedding_index), np.array(hash_index), np.array(pos), np.array(label), np.array(sentence_length), sentences
            

def prepare_ner(batch, trigram=False, step_length=ner_step_length, feature_length=ner_feature_length):

    embedding_index = []
    hash_index = []
    pos = []
    chunk = []
    label = []
    sentence_length = []
    sentences = []

    for sentence in batch:
        # sentence and sentence pos
        sequence = list(sentence[0])
        sequence_pos = list(sentence[1])
        sequence_chunk = list(sentence[2])

        # for trigram
        if trigram:
            # add start and end mark
            sequence.insert(0, '#')
            sequence.append('#')
            sequence_pos.insert(0, '#')
            sequence_pos.append('#')
            sequence_chunk.insert(0, '-X-')
            sequence_chunk.append('-X-')

        _embedding_index = [embedding_dict.get(each.strip().lower(), emb_vocab+1) for each in sequence]
        _hash_index = [chunk_hash_dict.get(each.strip().lower(), chunk_hash_vocab+1) for each in sequence]
        sentences.append(sentence[0])
        _pos = [ner_POS[each] for each in sequence_pos]
        _chunk = [ner_chunk[each] for each in sequence_chunk]
        _label = [IOB[each] for each in sentence[2]]
        length = len(_label)

        _label.extend([0]*(chunk_step_length-length))
        _embedding_index.extend([0]*(chunk_step_length-length))
        _hash_index.extend([0]*(chunk_step_length-length))

        embedding_index.append(_embedding_index)
        hash_index.append(_hash_index)
        pos.append(_pos)
        chunk.append(_chunk)
        label.append(_label)
        # record the sentence length for calculate accuracy
        sentence_length.append(length)

    return np.array(embedding_index), np.array(hash_index), np.array(pos), np.array(chunk), np.array(label), np.array(sentence_length), sentences


def prepare_additional(batch, step_length=step_length, feature_length=feature_length):
    special_case = re.compile(r'^[^a-zA-Z0-9]*$')
    lower_case = re.compile(r'^[a-z]*$')
    additional_feature = []
    sentence_length = []
    for sentence in batch:
        # sentence and sentence pos
        sequence = list(sentence[0])
        length = len(sequence)
        sentence_length.append(length)
        spelling_feature = np.zeros((length, additional_length))
        for i, word in enumerate(sequence):
            word = word.strip()
            spelling = np.zeros(additional_length)
            # is all letter is uppercase, digit or other
            # all uppercase
            if word.isupper():
                spelling[0] = 1
            # all lowercase
            elif re.match(lower_case, word):
                spelling[1] = 1
            # all digit
            elif word.isdigit():
                spelling[2] = 1
            # contain special character
            elif re.match(special_case, word):
                spelling[3] = 1
            # end with 's
            elif word=="'s":
                spelling[4] = 1
            else:
                spelling[5] = 1

            first_ele = word[0]
            # start with alpha
            if first_ele.isalpha():
                # start with upper
                if first_ele.isupper():
                    spelling[6] = 1
            # start with digit
            elif first_ele.isdigit():
                spelling[7] = 1
            else:
                spelling[8] = 1

            spelling_feature[i] = spelling
        additional_feature.append(spelling_feature)
    return np.array(additional_feature), np.array(sentence_length)
