#!/usr/bin/env python3
""" module containint function that creates a bag of words
    embedding matrix """
import numpy as np


def bag_of_words(sentences, vocb=None):
    """ function that creates a bag of words embedding matrix """

    # create vocab list if one does not exist
    if vocb is None:
        flag = 1
        vocb = []
    else:
        flag = 0

    split_sentences = []

    for i in range(len(sentences)):
        split_sentence = []

        # remove uppercase
        lower = sentences[i].lower()

        # tokenize sentence, remove punctuation
        doc_vocab = lower.split(' ')
        for token in doc_vocab:

            # remove any 's
            if token.endswith("'s"):
                token = token[:-2]
            token = ''.join(filter(lambda x: x.islower() or x.isspace(), token))

            #save cleaned split sentences
            split_sentence.append(token)

            # add token to vocab list if not already present
            if flag == 1:
                if token not in vocb:
                    vocb.append(token)
        split_sentences.append(split_sentence)
    if flag == 1:
        vocb = sorted(vocb)

    # create empty vectors
    embeddings = np.zeros((len(sentences), len(vocb)))

    # check each vocab word for presence in each sentence
    for x in range(embeddings.shape[1]):
        for y in range(embeddings.shape[0]):
            if vocb[x] in split_sentences[y]:
                embeddings[y][x] += 1
        
    return embeddings.astype(int), vocb