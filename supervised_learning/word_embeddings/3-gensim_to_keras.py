#!/usr/bin/env python3
"""" module containing function that convers a gensim word2vec model to a keras embedding layer """
import gensim as gm
import tensorflow.keras as K
import numpy as np


def gensim_to_keras(model):
    """ function that converts a gensim word2vec model to a keras embedding layer """
    sample = model.wv.index_to_key[0]
    rows = len(model.wv.index_to_key)
    cols = len(model.wv[sample])
    embedding_mat = np.zeros((rows, cols))
    i = 0
    for word in model.wv.index_to_key:
        embedding_mat[i] = model.wv[word]
        i += 1

    embedding_layer = K.layers.Embedding(rows, cols, weights=[embedding_mat], trainable=True)
    return embedding_layer