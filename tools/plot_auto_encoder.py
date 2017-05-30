import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import json


hash_embedding = pd.read_csv('../preprocessing/ner-auto-encoder-2/auto-encoder-embeddings.txt', delimiter=' ', header=None)
hash_embedding = hash_embedding.values

with open('../preprocessing/ner-auto-encoder-2/auto_encoder.json') as m:
    chunk_hash_dict = json.load(m)

embdding = []

morphology = ["ment", "tion", "ing", "ble"]
threshold = 100

for m in morphology:
    cnt = 0
    for word in chunk_hash_dict:
        # if m in word:
        if word.endswith(m):
            cnt += 1
            embdding.append(hash_embedding[chunk_hash_dict[word]-1].tolist())
        if cnt>=threshold:
            break

embdding = np.array(embdding)
print(embdding.shape)
pca=PCA(n_components=2, copy=True, whiten=False)
PCA_embedding = pca.fit_transform(embdding)
PCA_embedding = PCA_embedding.tolist()
coordinate = list(zip(*PCA_embedding))

color = ['b', 'r', 'y', 'g', 'c']
marker = ['x', '1', '+', 'o', 's']

for i in range(4):
    plt.scatter(coordinate[0][i*threshold:(i+1)*threshold], coordinate[1][i*threshold:(i+1)*threshold], marker=marker[i], c='k', )

plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.savefig("repre", dpi=300)
plt.show()