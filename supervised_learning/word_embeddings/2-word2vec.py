#!/usr/bin/env python3
""" module containing function that creates and trains a Word2Vec model """
import gensim as gm


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """ function that creates and trains a Word2Vec model """
    if cbow:
        sg = 0
    else:
        sg = 1
    word2vec = gm.models.Word2Vec(vector_size=size, min_count=min_count, window=window, negative=negative, sg=sg, seed=seed, workers=workers, epochs=iterations)
    word2vec.build_vocab(sentences)
    word2vec.train(sentences, total_examples=len(sentences), epochs=word2vec.epochs)
    return word2vec