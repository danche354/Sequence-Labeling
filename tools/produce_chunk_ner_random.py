import sys

import numpy as np
import pandas as pd

import conf


np.random.seed(0)


chunk_hash_vocab = conf.chunk_hash_vocab
chunk_hash_length = conf.chunk_hash_length

chunk_embedding = np.random.rand(chunk_hash_vocab, chunk_hash_length)
# chunk_embedding.astype(np.float32)

np.savetxt("../preprocessing/random/chunk_embeddings.txt", chunk_embedding, fmt="%.7f", delimiter=" ")


ner_hash_vocab = conf.ner_hash_vocab
ner_hash_length = conf.ner_hash_length

ner_embedding = np.random.rand(ner_hash_vocab, ner_hash_length)
# ner_embedding.dtype="np.float64"

np.savetxt("../preprocessing/random/ner_embeddings.txt", ner_embedding, fmt="%.7f", delimiter=" ")