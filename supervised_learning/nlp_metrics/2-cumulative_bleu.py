#!/usr/bin/env python3
""" module containing function that calculates the cumulative N-gram BLEU
    score for a sentence """
import numpy as np


def cumulative_bleu(references, sentences, n):
    """ function that calculates the cumulative N-gram BLEU score for a
    sentence """
    N = n
    bleus = []

    # for each n-gram size
    for n in range(1, N + 1):
        n_grams = []

        # create n-grams of the correct size for candidate sentence
        for i in range(len(sentences) - n + 1):
            n_grams.append(" ".join(sentences[i:i + n]))
        rn_grams = []

        # create n-grams for the reference sentences
        for reference in references:
            rn_gram = []
            for i in range(len(reference) - n + 1):
                rn_gram.append(" ".join(reference[i:i + n]))
            rn_grams.append(rn_gram)

        # get initial number of words in sentence
        num_candidate = len(sentences)

        maximums = []
        ref_lengths = []

        # filter out repeated words
        unique_candidate = list(set(n_grams))

        # for each word in candidate sentence, check how many tiems it appears
        # in the references
        for i in range(len(unique_candidate)):
            counts = []
            for j in range(len(rn_grams)):

                # get lengths of all reference sentences to see  if
                # brevity penalty needs to be applied
                if i == len(unique_candidate) - 1:
                    ref_lengths.append(len(references[j]))
                counts.append(rn_grams[j].count(unique_candidate[i]))
            maximums.append(max(counts))

        # find inital score without brevity penalty
        bleu = (sum(maximums) / len(n_grams))
        bleu = np.log(bleu) * (1 / N)
        bleus.append(bleu)

    bleu = np.exp(sum(bleus))

    # apply BP if needed
    if num_candidate <= min(ref_lengths):
        bleu *= np.exp(1 - min(ref_lengths) / num_candidate)

    return bleu
