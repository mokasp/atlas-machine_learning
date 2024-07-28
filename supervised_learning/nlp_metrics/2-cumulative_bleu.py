#!/usr/bin/env python3
""" module containing function that calculates the cumulative N-gram BLEU
    score for a sentence """
import numpy as np


def cumulative_bleu(references, sentences, n):
    """ function that calculates the cumulative N-gram BLEU score for a
    sentence """
    N = n
    bleus = []
    for n in range(1, N + 1):
        n_grams = []
        for i in range(len(sentences) - n + 1):
            n_grams.append(" ".join(sentences[i:i + n]))
        rn_grams = []
        for reference in references:
            rn_gram = []
            for i in range(len(reference) - n + 1):
                rn_gram.append(" ".join(reference[i:i + n]))
            rn_grams.append(rn_gram)

        num_candidate = len(sentences)
        maximums = []
        ref_lengths = []
        unique_candidate = list(set(n_grams))
        for i in range(len(unique_candidate)):
            counts = []
            for j in range(len(rn_grams)):
                if i == len(unique_candidate) - 1:
                    ref_lengths.append(len(references[j]))
                counts.append(rn_grams[j].count(unique_candidate[i]))
            maximums.append(max(counts))

        bleu = (sum(maximums) / len(n_grams))
        bleu = np.log(bleu) * (1 / N)
        bleus.append(bleu)

    bleu = np.exp(sum(bleus))

    if num_candidate <= min(ref_lengths):
        bleu *= np.exp(1 - min(ref_lengths) / num_candidate)

    return bleu
