#!/usr/bin/env python3
""" implements a semantic search function using a pre-trained Universal
    Sentence Encoder. It allows users to input a query and find the most
    semantically similar document from a corpus of reference documents.

    Dependencies:
        - tensorflow_hub: Provides access to pre-trained models, including the
            Universal Sentence Encoder.
        - numpy: Used for numerical operations, particularly in similarity
                 scoring.
        - os: Used for file and directory management.

    Modules:
        - semantic_search(corpus_path, sentence): Identifies the most
                semantically similar document from a corpus to a given
                query sentence.

    Usage:
        1. Prepare a directory containing text files (the corpus) that the
           system will use as reference documents.
        2. Call the `semantic_search` function, providing the path to the
           corpus directory and the query sentence.
        3. The function will return the most semantically similar document
           from the corpus.
"""
import tensorflow_hub as th
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    """ finds the most semantically similar document to the given query using
        the Universal Sentence Encoder.

        Args:
            corpus_path (str): The path to the directory containing reference
                           documents.
            sentence (str): The query sentence to search for in the corpus.

        Returns:
            str: The most semantically similar document to the query.

        Process:
            1. Collect all reference documents from the specified directory.
            2. Load the Universal Sentence Encoder model.
            3. Encode the query sentence and reference documents.
            4. Compute the similarity score between the query and each
               document.
            5. Return the document with the highest similarity score.
    """

    # collect all reference documents
    references = []
    for file in os.listdir(corpus_path):
        if not file.startswith('.'):
            with open(corpus_path + '/' + file) as f:
                references.append(f.read())

    # load the sentence encoder
    sentence_encoder = th.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/4")
    # combine the query with the references and encode this vector
    references = [sentence] + references
    outputs = sentence_encoder(references)['outputs']

    # find similarity score of each document to get most similar doc
    similarity_score = np.inner(outputs, outputs)[0][1:]
    similar_idx = np.argmax(similarity_score) + 1
    similar_doc = references[similar_idx]
    return similar_doc
