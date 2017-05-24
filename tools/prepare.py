
import numpy as np

import re

import hashing
import conf

embedding_dict = conf.senna_dict
emb_vocab = conf.senna_vocab

chunk_hash_dict = conf.chunk_hash_dict
chunk_hash_dict_2 = conf.chunk_hash_dict_2
chunk_hash_vocab = conf.chunk_hash_vocab

ner_hash_dict = conf.ner_hash_dict
ner_hash_dict_2 = conf.ner_hash_dict_2
ner_hash_vocab = conf.ner_hash_vocab

chunk_step_length = conf.chunk_step_length

ner_step_length = conf.ner_step_length

chunk_ALL_IOB = conf.chunk_ALL_IOB_encode
chunk_NP_IOB = conf.chunk_NP_IOB_encode
chunk_POS = conf.chunk_POS_encode


ner_POS = conf.ner_POS_encode
ner_chunk = conf.ner_chunk_encode
ner_IOB = conf.ner_IOB_encode
ner_BIOES = conf.ner_BIOES_encode

additional_length = conf.additional_length
gazetteer_length = conf.gazetteer_length
BIOES_gazetteer_length = conf.BIOES_gazetteer_length


def prepare_auto_encoder(batch, task='chunk', gram='tri'):
    word_hashing = hashing.sen2matrix(batch, task, gram)
    return word_hashing

def prepare_chunk(batch, trigram=False, gram='tri', chunk_type='NP', step_length=chunk_step_length):
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
        if gram=='tri':
            _hash_index = [chunk_hash_dict.get(each.strip().lower(), chunk_hash_vocab+1) for each in sequence]
        elif gram=='bi':
            _hash_index = [chunk_hash_dict_2.get(each.strip().lower(), chunk_hash_vocab+1) for each in sequence]
        sentences.append(sentence[0])
        _pos = [chunk_POS[each] for each in sequence_pos]
        _label = [IOB[each] for each in sentence[2]]
        length = len(_label)

        _label.extend([0]*(step_length-length))
        _embedding_index.extend([0]*(step_length-length))
        _hash_index.extend([0]*(step_length-length))

        embedding_index.append(_embedding_index)
        hash_index.append(_hash_index)
        pos.append(_pos)
        label.append(_label)
        # record the sentence length for calculate accuracy
        sentence_length.append(length)

    return np.array(embedding_index), np.array(hash_index), np.array(pos), np.array(label), np.array(sentence_length), sentences

# raw hashing
def prepare_chunk_raw(batch, trigram=False, gram='tri', chunk_type='NP', step_length=chunk_step_length):
    if chunk_type=='ALL':
        IOB = chunk_ALL_IOB
    else:
        IOB = chunk_NP_IOB

    embedding_index = []
    hash_representation = []
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
        _hash_representation = hashing.sen2matrix(sequence, task="ner", gram=gram)
        sentences.append(sentence[0])
        _pos = [chunk_POS[each] for each in sequence_pos]
        _label = [IOB[each] for each in sentence[2]]
        length = len(_label)

        _label.extend([0]*(step_length-length))
        _embedding_index.extend([0]*(step_length-length))

        embedding_index.append(_embedding_index)
        hash_representation.append(_hash_representation)
        pos.append(_pos)
        label.append(_label)
        # record the sentence length for calculate accuracy
        sentence_length.append(length)

    return np.array(embedding_index), np.array(hash_representation), np.array(pos), np.array(label), np.array(sentence_length), sentences
                      

# raw hashing
def prepare_ner_raw(batch, trigram=False, gram='tri', form='BIO', step_length=ner_step_length):
    embedding_index = []
    hash_representation = []
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

        _embedding_index = [embedding_dict.get(each.strip().lower(), emb_vocab+1) for each in sequence]
        _hash_representation = hashing.sen2matrix(sequence, task="ner", gram=gram)
        sentences.append(sentence[0])
        _pos = [ner_POS[each] for each in sequence_pos]
        _chunk = [ner_chunk[each] for each in sequence_chunk]
        if form=="BIO":
            _label = [ner_IOB[each] for each in sentence[3]]
        elif form=="BIOES":
            _label = [ner_BIOES[each] for each in sentence[3]]
        length = len(_label)

        _label.extend([0]*(step_length-length))
        _embedding_index.extend([0]*(step_length-length))

        embedding_index.append(_embedding_index)
        hash_representation.append(_hash_representation)
        pos.append(_pos)
        chunk.append(_chunk)
        label.append(_label)
        # record the sentence length for calculate accuracy
        sentence_length.append(length)

    return np.array(embedding_index), np.array(hash_representation), np.array(pos), np.array(chunk), np.array(label), np.array(sentence_length), sentences
                      

def prepare_ner(batch, form='BIO', trigram=False, gram='tri', step_length=ner_step_length):

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
        if gram=='tri':
            _hash_index = [ner_hash_dict.get(each.strip().lower(), ner_hash_vocab+1) for each in sequence]
        elif gram=='bi':
            _hash_index = [ner_hash_dict_2.get(each.strip().lower(), ner_hash_vocab+1) for each in sequence]
        sentences.append(sentence[0])
        _pos = [ner_POS[each] for each in sequence_pos]
        _chunk = [ner_chunk[each] for each in sequence_chunk]
        if form=="BIO":
            _label = [ner_IOB[each] for each in sentence[3]]
        elif form=="BIOES":
            _label = [ner_BIOES[each] for each in sentence[3]]
        length = len(_label)

        _label.extend([0]*(step_length-length))
        _embedding_index.extend([0]*(step_length-length))
        _hash_index.extend([0]*(step_length-length))

        embedding_index.append(_embedding_index)
        hash_index.append(_hash_index)
        pos.append(_pos)
        chunk.append(_chunk)
        label.append(_label)
        # record the sentence length for calculate accuracy
        sentence_length.append(length)

    return np.array(embedding_index), np.array(hash_index), np.array(pos), np.array(chunk), np.array(label), np.array(sentence_length), sentences


def prepare_gazetteer(batch, gazetteer='senna'):
    if gazetteer == 'senna':
        LOC = conf.LOC
        PER = conf.PER
        ORG = conf.ORG
        MISC = conf.MISC
    elif gazetteer == 'conll':
        LOC = conf.LOC_conll
        PER = conf.PER_conll
        ORG = conf.ORG_conll
        MISC = conf.MISC_conll
    step_length = ner_step_length
    gazetteer_feature = []
    sentence_length = []
    for sentence in batch:
        sequence = list(sentence[0])
        sequence = [each.strip().lower() for each in sequence]
        length = len(sequence)
        sentence_length.append(length)
        _gazetteer_feature = np.zeros((length, gazetteer_length))
        for i, word in enumerate(sequence):
            gazetteer = np.zeros(gazetteer_length)
            if word in LOC:
                gazetteer[0] = 1
            if word in ORG:
                gazetteer[1] = 1
            if word in PER:
                gazetteer[2] = 1
            if word in MISC:
                gazetteer[3] = 1
            _gazetteer_feature[i] = gazetteer
        
        for i in range(length-1):
            flag = False
            word = sequence[i] + " " + sequence[i+1]
            gazetteer = np.zeros((2, gazetteer_length))
            if word in LOC:
                gazetteer[:,0] = 1
                flag = True
            if word in ORG:
                gazetteer[:,1] = 1
                flag = True
            if word in PER:
                gazetteer[:,2] = 1
                flag = True
            if word in MISC:
                gazetteer[:,3] = 1
                flag = True
            if flag:
                _gazetteer_feature[i:i+2] = gazetteer


        for i in range(length-2):
            flag = False
            word = sequence[i] + " " + sequence[i+1] + " " + sequence[i+2]
            gazetteer = np.zeros((3, gazetteer_length))
            if word in LOC:
                gazetteer[:,0] = 1
                flag = True
            if word in ORG:
                gazetteer[:,1] = 1
                flag = True
            if word in PER:
                gazetteer[:,2] = 1
                flag = True
            if word in MISC:
                gazetteer[:,3] = 1
                flag = True
            if flag:
                _gazetteer_feature[i:i+3] = gazetteer
            
        for i in range(length-3):
            flag = False
            word = sequence[i] + " " + sequence[i+1] + " " + sequence[i+2] + " "  + sequence[i+3]
            gazetteer = np.zeros((4, gazetteer_length))
            if word in LOC:
                gazetteer[:,0] = 1
                flag = True
            if word in ORG:
                gazetteer[:,1] = 1
                flag = True
            if word in PER:
                gazetteer[:,2] = 1
                flag = True
            if word in MISC:
                gazetteer[:,3] = 1
                flag = True
            if flag:
                _gazetteer_feature[i:i+4] = gazetteer
        
        for i in range(length-4):
            flag = False
            word = sequence[i] + " "  + sequence[i+1] + " " + sequence[i+2] + " "  + sequence[i+3] + " " + sequence[i+4]
            gazetteer = np.zeros((5, gazetteer_length))
            if word in LOC:
                gazetteer[:,0] = 1
                flag = True
            if word in ORG:
                gazetteer[:,1] = 1
                flag = True
            if word in PER:
                gazetteer[:,2] = 1
                flag = True
            if word in MISC:
                gazetteer[:,3] = 1
                flag = True
            if flag:
                _gazetteer_feature[i:i+5] = gazetteer
        
        gazetteer_feature.append(_gazetteer_feature)
    return np.array(gazetteer_feature), np.array(sentence_length)

def prepare_gazetteer_BIOES(batch, gazetteer='senna'):
    if gazetteer == 'senna':
        LOC = conf.LOC
        PER = conf.PER
        ORG = conf.ORG
        MISC = conf.MISC
    elif gazetteer == 'conll':
        LOC = conf.LOC_conll
        PER = conf.PER_conll
        ORG = conf.ORG_conll
        MISC = conf.MISC_conll
    step_length = ner_step_length
    gazetteer_feature = []
    sentence_length = []
    for sentence in batch:
        sequence = list(sentence[0])
        sequence = [each.strip().lower() for each in sequence]
        length = len(sequence)
        sentence_length.append(length)
        _gazetteer_feature = np.zeros((length, BIOES_gazetteer_length))

        i = 0
        while (i<length):
            word = sequence[i]
            gazetteer = np.zeros(BIOES_gazetteer_length)
            if word in LOC:
                gazetteer[3] = 1
            if word in ORG:
                gazetteer[7] = 1
            if word in PER:
                gazetteer[11] = 1
            if word in MISC:
                gazetteer[15] = 1
            _gazetteer_feature[i] = gazetteer
            i = i+1

        i = 0
        flag = False
        while (i<length-1):
            word = sequence[i] + " " + sequence[i+1]
            gazetteer = np.zeros((2, BIOES_gazetteer_length))
            if word in LOC:
                gazetteer[0,0] = 1
                gazetteer[1,2] = 1
                flag = True
            if word in ORG:
                gazetteer[0,4] = 1
                gazetteer[1,6] = 1
                flag = True
            if word in PER:
                gazetteer[0,8] = 1
                gazetteer[1,10] = 1
                flag = True
            if word in MISC:
                gazetteer[0,12] = 1
                gazetteer[1,14] = 1
                flag = True
            if flag:
                _gazetteer_feature[i:i+2] = gazetteer
                i = i+2
                flag = False
            else:
                i = i+1

        i = 0
        flag = False
        while (i<length-2):
            word = sequence[i] + " " + sequence[i+1] + " " + sequence[i+2]
            gazetteer = np.zeros((3, BIOES_gazetteer_length))
            if word in LOC:
                gazetteer[0,0] = 1
                gazetteer[1,1] = 1
                gazetteer[2,2] = 1
                flag = True
            if word in ORG:
                gazetteer[0,4] = 1
                gazetteer[1,5] = 1
                gazetteer[2,6] = 1
                flag = True
            if word in PER:
                gazetteer[0,8] = 1
                gazetteer[1,9] = 1
                gazetteer[2,10] = 1
                flag = True
            if word in MISC:
                gazetteer[0,12] = 1
                gazetteer[1,13] = 1
                gazetteer[2,14] = 1
                flag = True
            if flag:
                _gazetteer_feature[i:i+3] = gazetteer
                i = i+3
                flag = False
            else:
                i = i+1

        i = 0
        flag = False
        while (i<length-3):
            word = sequence[i] + " " + sequence[i+1] + " " + sequence[i+2] + " "  + sequence[i+3]
            gazetteer = np.zeros((4, BIOES_gazetteer_length))
            if word in LOC:
                gazetteer[0,0] = 1
                gazetteer[1:3,1] = 1
                gazetteer[3,2] = 1
                flag = True
            if word in ORG:
                gazetteer[0,4] = 1
                gazetteer[1:3,5] = 1
                gazetteer[3,6] = 1
                flag = True
            if word in PER:
                gazetteer[0,8] = 1
                gazetteer[1:3,9] = 1
                gazetteer[3,10] = 1
                flag = True
            if word in MISC:
                gazetteer[0,12] = 1
                gazetteer[1:3,13] = 1
                gazetteer[3,14] = 1
                flag = True
            if flag:
                _gazetteer_feature[i:i+4] = gazetteer
                i = i+4
                flag = False
            else:
                i = i+1

        i = 0
        flag = False
        while (i<length-4):
            word = sequence[i] + " "  + sequence[i+1] + " " + sequence[i+2] + " "  + sequence[i+3] + " " + sequence[i+4]
            gazetteer = np.zeros((5, BIOES_gazetteer_length))
            if word in LOC:
                gazetteer[0,0] = 1
                gazetteer[1:4,1] = 1
                gazetteer[4,2] = 1
                flag = True
            if word in ORG:
                gazetteer[0,4] = 1
                gazetteer[1:4,5] = 1
                gazetteer[4,6] = 1
                flag = True
            if word in PER:
                gazetteer[0,8] = 1
                gazetteer[1:4,9] = 1
                gazetteer[4,10] = 1
                flag = True
            if word in MISC:
                gazetteer[0,12] = 1
                gazetteer[1:4,13] = 1
                gazetteer[4,14] = 1
                flag = True
            if flag:
                _gazetteer_feature[i:i+5] = gazetteer
                i = i+5
                flag = False
            else:
                i = i+1
        gazetteer_feature.append(_gazetteer_feature)
    return np.array(gazetteer_feature), np.array(sentence_length)


    
def prepare_additional(batch, task='chunk'):
    if task=='chunk':
        step_length = chunk_step_length
    elif task=='ner':
        step_length = ner_step_length
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


def gazetteer_lookup(batch, chunktag, data, gazetteer='senna'):
    if data=="test":
        if gazetteer == 'senna':
            LOC = conf.LOC
            PER = conf.PER
            ORG = conf.ORG
            MISC = conf.MISC
        elif gazetteer == 'conll':
            LOC = conf.LOC_conll
            PER = conf.PER_conll
            ORG = conf.ORG_conll
            MISC = conf.MISC_conll

        sequence = [each.strip().lower() for each in batch]
        length = len(sequence)
        for i, word in enumerate(sequence):
            print(word)
            if word in LOC:
                chunktag[i] = "I-LOC"
            elif word in ORG:
                chunktag[i] = "I-ORG"
            elif word in PER:
                chunktag[i] = "I-PER"
            elif word in MISC:
                chunktag[i] = "I-MISC"

        for i in range(length-1):
            word = sequence[i] + " " + sequence[i+1]
            if word in LOC:
                chunktag[i] = "B-LOC"
                chunktag[i+1] = "I-LOC"
            elif word in ORG:
                chunktag[i] = "B-ORG"
                chunktag[i+1] = "I-ORG"
            elif word in PER:
                chunktag[i] = "B-PER"
                chunktag[i+1] = "I-PER"
            elif word in MISC:
                chunktag[i] = "B-MISC"
                chunktag[i+1] = "I-MISC"


        for i in range(length-2):
            word = sequence[i] + " " + sequence[i+1] + " " + sequence[i+2]
            if word in LOC:
                chunktag[i] = "B-LOC"
                chunktag[i+1] = "I-LOC"
                chunktag[i+2] = "I-LOC"
            elif word in ORG:
                chunktag[i] = "B-ORG"
                chunktag[i+1] = "I-ORG"
                chunktag[i+2] = "I-ORG"
            elif word in PER:
                chunktag[i] = "B-PER"
                chunktag[i+1] = "I-PER"
                chunktag[i+2] = "I-PER"
            elif word in MISC:
                chunktag[i] = "B-MISC"
                chunktag[i+1] = "I-MISC"
                chunktag[i+2] = "I-MISC"

            
        for i in range(length-3):
            word = sequence[i] + " " + sequence[i+1] + " " + sequence[i+2] + " "  + sequence[i+3]
            if word in LOC:
                chunktag[i] = "B-LOC"
                chunktag[i+1] = "I-LOC"
                chunktag[i+2] = "I-LOC"
                chunktag[i+3] = "I-LOC"
            elif word in ORG:
                chunktag[i] = "B-ORG"
                chunktag[i+1] = "I-ORG"
                chunktag[i+2] = "I-ORG"
                chunktag[i+3] = "I-ORG"
            elif word in PER:
                chunktag[i] = "B-PER"
                chunktag[i+1] = "I-PER"
                chunktag[i+2] = "I-PER"
                chunktag[i+3] = "I-PER"
            elif word in MISC:
                chunktag[i] = "B-MISC"
                chunktag[i+1] = "I-MISC"
                chunktag[i+2] = "I-MISC"
                chunktag[i+3] = "I-MISC"


        for i in range(length-4):
            word = sequence[i] + " "  + sequence[i+1] + " " + sequence[i+2] + " "  + sequence[i+3] + " " + sequence[i+4]
            if word in LOC:
                chunktag[i] = "B-LOC"
                chunktag[i+1] = "I-LOC"
                chunktag[i+2] = "I-LOC"
                chunktag[i+3] = "I-LOC"
                chunktag[i+4] = "I-LOC"
            elif word in ORG:
                chunktag[i] = "B-ORG"
                chunktag[i+1] = "I-ORG"
                chunktag[i+2] = "I-ORG"
                chunktag[i+3] = "I-ORG"
                chunktag[i+4] = "I-ORG"
            elif word in PER:
                chunktag[i] = "B-PER"
                chunktag[i+1] = "I-PER"
                chunktag[i+2] = "I-PER"
                chunktag[i+3] = "I-PER"
                chunktag[i+4] = "I-PER"
            elif word in MISC:
                chunktag[i] = "B-MISC"
                chunktag[i+1] = "I-MISC"
                chunktag[i+2] = "I-MISC"
                chunktag[i+3] = "I-MISC"
                chunktag[i+4] = "I-MISC"
    elif data=="dev":
        if gazetteer == 'senna':
            LOC = conf.LOC
            PER = conf.PER
            ORG = conf.ORG
            MISC = conf.MISC
        elif gazetteer == 'conll':
            LOC = conf.LOC_conll
            PER = conf.PER_conll
            ORG = conf.ORG_conll
            MISC = conf.MISC_conll

        sequence = [each.strip().lower() for each in batch]
        length = len(sequence)
        for i, word in enumerate(sequence):
            print(word)
            if word in LOC:
                chunktag[i] = "S-LOC"
            elif word in ORG:
                chunktag[i] = "S-ORG"
            elif word in PER:
                chunktag[i] = "S-PER"
            elif word in MISC:
                chunktag[i] = "S-MISC"

        for i in range(length-1):
            word = sequence[i] + " " + sequence[i+1]
            if word in LOC:
                chunktag[i] = "B-LOC"
                chunktag[i+1] = "E-LOC"
            elif word in ORG:
                chunktag[i] = "B-ORG"
                chunktag[i+1] = "E-ORG"
            elif word in PER:
                chunktag[i] = "B-PER"
                chunktag[i+1] = "E-PER"
            elif word in MISC:
                chunktag[i] = "B-MISC"
                chunktag[i+1] = "E-MISC"


        for i in range(length-2):
            word = sequence[i] + " " + sequence[i+1] + " " + sequence[i+2]
            if word in LOC:
                chunktag[i] = "B-LOC"
                chunktag[i+1] = "I-LOC"
                chunktag[i+2] = "E-LOC"
            elif word in ORG:
                chunktag[i] = "B-ORG"
                chunktag[i+1] = "I-ORG"
                chunktag[i+2] = "E-ORG"
            elif word in PER:
                chunktag[i] = "B-PER"
                chunktag[i+1] = "I-PER"
                chunktag[i+2] = "E-PER"
            elif word in MISC:
                chunktag[i] = "B-MISC"
                chunktag[i+1] = "I-MISC"
                chunktag[i+2] = "E-MISC"

            
        for i in range(length-3):
            word = sequence[i] + " " + sequence[i+1] + " " + sequence[i+2] + " "  + sequence[i+3]
            if word in LOC:
                chunktag[i] = "B-LOC"
                chunktag[i+1] = "I-LOC"
                chunktag[i+2] = "I-LOC"
                chunktag[i+3] = "E-LOC"
            elif word in ORG:
                chunktag[i] = "B-ORG"
                chunktag[i+1] = "I-ORG"
                chunktag[i+2] = "I-ORG"
                chunktag[i+3] = "E-ORG"
            elif word in PER:
                chunktag[i] = "B-PER"
                chunktag[i+1] = "I-PER"
                chunktag[i+2] = "I-PER"
                chunktag[i+3] = "E-PER"
            elif word in MISC:
                chunktag[i] = "B-MISC"
                chunktag[i+1] = "I-MISC"
                chunktag[i+2] = "I-MISC"
                chunktag[i+3] = "E-MISC"


        for i in range(length-4):
            word = sequence[i] + " "  + sequence[i+1] + " " + sequence[i+2] + " "  + sequence[i+3] + " " + sequence[i+4]
            if word in LOC:
                chunktag[i] = "B-LOC"
                chunktag[i+1] = "I-LOC"
                chunktag[i+2] = "I-LOC"
                chunktag[i+3] = "I-LOC"
                chunktag[i+4] = "E-LOC"
            elif word in ORG:
                chunktag[i] = "B-ORG"
                chunktag[i+1] = "I-ORG"
                chunktag[i+2] = "I-ORG"
                chunktag[i+3] = "I-ORG"
                chunktag[i+4] = "E-ORG"
            elif word in PER:
                chunktag[i] = "B-PER"
                chunktag[i+1] = "I-PER"
                chunktag[i+2] = "I-PER"
                chunktag[i+3] = "I-PER"
                chunktag[i+4] = "E-PER"
            elif word in MISC:
                chunktag[i] = "B-MISC"
                chunktag[i+1] = "I-MISC"
                chunktag[i+2] = "I-MISC"
                chunktag[i+3] = "I-MISC"
                chunktag[i+4] = "E-MISC"
    return chunktag
