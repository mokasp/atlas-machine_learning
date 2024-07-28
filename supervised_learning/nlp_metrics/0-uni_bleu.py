#!/usr/bin/env python3
""" module containing function that calculates the unigram BLEU score for a
    sentence """
import numpy as np


def uni_bleu(references, sentences):
    """ function that calculates the unigram BLEU score for a sentence """
    num_candidate = len(sentences)
    maximums = []
    ref_lengths = []
    unique_candidate = list(set(sentences))
    for i in range(len(unique_candidate)):
        counts = []
        for ref_sentence in references:
            if i == len(unique_candidate) - 1:
                ref_lengths.append(len(ref_sentence))
            counts.append(ref_sentence.count(unique_candidate[i]))
        maximums.append(max(counts))

    return (sum(maximums) / num_candidate) * np.exp(1 - min(ref_lengths) / num_candidate)