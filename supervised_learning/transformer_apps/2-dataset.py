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
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
    """ a class used to load, preprocess, and tokenize a dataset for machine
        translation.

        Attributes:
            tokenizer_en: A tokenizer object for the English language, used to
                          convert English sentences into sequences of subword
                          tokens.
            tokenizer_pt: A tokenizer object for the Portuguese language, used
                          to convert Portuguese sentences into sequences of
                          subword tokens.
            data_train: A TensorFlow dataset containing the preprocessed and
                        tokenized
                        training data, mapped with the `tf_encode` method.
            data_valid: A TensorFlow dataset containing the preprocessed and
                        tokenized validation data, mapped with the `tf_encode`
                        method.

        Methods:
            __init__(self):
                Initializes the Dataset class by loading the dataset,
                tokenizing it, and preparing the training and validation
                data for use.

            load_dataset(self):
                Loads the Portuguese to English translation dataset and
                returns the training and validation datasets.

            tokenize_dataset(self, data):
                Creates subword tokenizers for both the Portuguese and
                English languages using the training dataset.

            tf_encode(self, pt, en):
                TensorFlow wrapper around the `encode` method to allow
                for efficient preprocessing within the data pipeline.
    """

    def __init__(self):
        data_train, data_valid = self.load_dataset()
        tokenizer_en, tokenizer_pt = self.tokenize_dataset(data_train)
        self.tokenizer_en = tokenizer_en
        self.tokenizer_pt = tokenizer_pt
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

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

    def tokenize_dataset(self, data):
        """ creates subword tokenizers for both the Portuguese and English
            languages using the training dataset.

            Returns:
                tuple: A tuple containing the English and Portuguese
                  tokenizers. These tokenizers are capable of converting text
                  into subword tokens and back.
        """
        token_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2**15)
        token_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2**15)
        return token_en, token_pt

    def tf_encode(self, pt, en):
        """ TensorFlow wrapper around the `encode` method to allow for
            efficient preprocessing within the data pipeline.

            Args:
                pt (tf.Tensor): The Portuguese sentence.
                en (tf.Tensor): The English sentence.

            Returns:
                tuple: A tuple containing the encoded Portuguese and English
                       sentences
        """

        def encode(pt, en):
            pt_size = [self.tokenizer_pt.vocab_size]
            en_size = [self.tokenizer_en.vocab_size]
            pt_tokens = pt_size + self.tokenizer_pt.encode(
                pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
            en_tokens = en_size + self.tokenizer_en.encode(
                en.numpy()) + [self.tokenizer_en.vocab_size + 1]
            return pt_tokens, en_tokens

        result_pt, result_en = tf.py_function(
            lambda pt, en: encode(
                pt, en), [
                pt, en], [
                tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
