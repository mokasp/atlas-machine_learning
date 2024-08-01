#!/usr/bin/env python3
""" module containing function that calculates the unigram BLEU score for a
    sentence """
import numpy as np


def uni_bleu(references, sentences):
    """ function that calculates the unigram BLEU score for a sentence """
    # get initial number of words in sentence
    num_candidate = len(sentences)

    maximums = []
    ref_lengths = []

    # filter out repeated words
    unique_candidate = list(set(sentences))

    # for each word in candidate sentence, check how many tiems it appears
    # in the references
    for i in range(len(unique_candidate)):
        counts = []
        for ref_sentence in references:

            # get lengths of all reference sentences to see if
            # brevity penalty needs to be applied
            if i == len(unique_candidate) - 1:
                ref_lengths.append(len(ref_sentence))
            counts.append(ref_sentence.count(unique_candidate[i]))
        maximums.append(max(counts))

    # find inital score without brevity penalty
    bleu = (sum(maximums) / num_candidate)

    # apply BP if needed
    if num_candidate <= min(ref_lengths):
        bleu *= np.exp(1 - min(ref_lengths) / num_candidate)

    return bleu
