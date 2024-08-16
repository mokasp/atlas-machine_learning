#!/usr/bin/env python3
from transformers import BertTokenizer, BertModel
import tensorflow as tf
import tensorflow_hub as th
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    
    # collect all reference documents
    references = []
    for file in os.listdir(corpus_path):
        if not file.startswith('.'):
            with open(corpus_path + '/' + file) as f:
                references.append(f.read())
    
    # load the sentence encoder
    sentence_encoder = th.load("https://tfhub.dev/google/universal-sentence-encoder-large/4")
    # combine the query with the references and encode this vector
    references = [sentence] + references
    outputs = sentence_encoder(references)['outputs']
    
    # find similarity score of each document to get most similar doc
    similarity_score = np.inner(outputs, outputs)[0][1:]
    similar_idx = np.argmax(similarity_score) + 1
    similar_doc = references[similar_idx]
    return similar_doc