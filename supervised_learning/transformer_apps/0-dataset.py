#!/usr/bin/env python3
""" this script defines a `Dataset` class that represents a loaded and
    preprocessed dataset to be used for machine translation tasks. The
    class handles the loading of training and validation datasets, as well
    as tokenization of the input text in both the source and target languages.

    Dependencies:
        - tensorflow_datasets: A collection of ready-to-use datasets with
            TensorFlow for machine learning and artificial intelligence
            applications.

    Class:
        - Dataset: A class that encapsulates the process of loading and
            preparing a dataset for machine translation, specifically
            Portuguese to English translation using the TED Talks dataset.

    Usage:
        1. Instantiate the `Dataset` class to load the data and perform
           tokenization.
        2. Access the training and validation datasets via `data_train` and
           `data_valid`.
        3. Access the tokenizers via `tokenizer_en` and `tokenizer_pt`.
"""
import tensorflow_datasets as tfds


class Dataset():
    """ a class used to load, preprocess, and tokenize a dataset for machine
        translation.

        Attributes:
            data_train: A TensorFlow dataset containing the training data,
                which consists of pairs of Portuguese sentences and their
                English translations.
            data_valid: A TensorFlow dataset containing the validation data,
                which consists of pairs of Portuguese sentences and their
                English translations.
            tokenizer_en: A tokenizer object for the English language, used to
                convert English sentences into sequences of subword tokens.
            tokenizer_pt: A tokenizer object for the Portuguese language, used
                to convert Portuguese sentences into sequences of subword
                tokens.
    """

    def __init__(self):
        """ initializes the Dataset class by loading the dataset and creating
            tokenizers for both Portuguese and English languages. """
        data_train, data_valid = self.load_dataset()
        self.data_train = data_train
        self.data_valid = data_valid
        tokenizer_en, tokenizer_pt = self.tokenize_dataset()
        self.tokenizer_en = tokenizer_en
        self.tokenizer_pt = tokenizer_pt

    def load_dataset(self):
        """ loads the Portuguese to English translation dataset.

            Returns:
                tuple: A tuple containing the training and validation datasets.
                  Each dataset is a TensorFlow dataset of (Portuguese, English)
                  sentence pairs.
            """
        pt2en_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True)
        pt2en_val = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True)
        return pt2en_train, pt2en_val

    def tokenize_dataset(self):
        """ creates subword tokenizers for both the Portuguese and English
            languages using the training dataset.

            Returns:
                tuple: A tuple containing the English and Portuguese
                  tokenizers. These tokenizers are capable of converting text
                  into subword tokens and back.
        """
        token_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in self.data_train), target_vocab_size=2**15)
        token_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in self.data_train), target_vocab_size=2**15)
        return token_en, token_pt
