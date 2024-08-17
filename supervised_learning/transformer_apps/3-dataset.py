#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
    """ a class used to load, preprocess, and tokenize a dataset for machine
        translation.

        Attributes:
            batch_size (int): The batch size to be used for batching the
                datasets.
            max_len (int): The maximum sequence length for filtering
                sentences.
            tokenizer_en (SubwordTextEncoder): A tokenizer object for the
                English language, used to convert English sentences into
                sequences of subword tokens.
            tokenizer_pt (SubwordTextEncoder): A tokenizer object for the
                Portuguese language, used to convert Portuguese sentences
                into sequences of subword tokens.
            data_train (tf.data.Dataset): A TensorFlow dataset containing
                the preprocessed and tokenized training data, mapped with
                the `tf_encode` method.
            data_valid (tf.data.Dataset): A TensorFlow dataset containing the
                preprocessed and tokenized validation data, mapped with the
                `tf_encode` method.

        Methods:
            __init__(self, batch_size, max_len):
                Initializes the Dataset class by loading the dataset, creating
                tokenizers, and preparing the training and validation data
                with the specified batch size and maximum length.

            load_dataset(self):
                Loads the Portuguese to English translation dataset and
                returns the training and validation datasets.

            tokenize_dataset(self, data):
                Creates subword tokenizers for both the Portuguese and
                English languages using the provided training dataset.

            tf_encode(self, pt, en):
                TensorFlow wrapper around the `encode` method to allow for
                efficient preprocessing within the data pipeline.

            filter_by_length(self, pt, en):
                Filters out sentence pairs where either sentence exceeds the
                maximum length.

            filter_wrapper(self, pt, en):
                Wrapper function for filtering sentences by length, compatible
                with TensorFlow's `tf.data` API.
    """

    def __init__(self, batch_size, max_len):
        """ initializes the Dataset class by loading the dataset, creating
            tokenizers, and preparing the training and validation data with
            the specified batch size and maximum length.

            Args:
                batch_size (int): The batch size to be used for batching the
                    datasets.
                max_len (int): The maximum sequence length for filtering
                    sentences.
        """
        data_train, data_valid = self.load_dataset()
        tokenizer_en, tokenizer_pt = self.tokenize_dataset(
            (data_train, data_valid))
        self.tokenizer_en = tokenizer_en
        self.tokenizer_pt = tokenizer_pt
        self.max_len = max_len
        self.batch_size = batch_size
        self.data_train = data_train.map(
            self.tf_encode,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).filter(
            self.filter_wrapper).cache().shuffle(
            buffer_size=len(data_train)).padded_batch(batch_size)
        self.data_valid = data_valid.map(
            self.tf_encode).filter(
            self.filter_wrapper).padded_batch(batch_size)

    def load_dataset(self):
        """ loads the Portuguese to English translation dataset.

            Returns:
                tuple: A tuple containing the training and validation datasets
                        Each dataset is a TensorFlow dataset of (Portuguese,
                        English) sentence pairs.
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
            languages using the provided training dataset.

            Args:
                data (tuple): A tuple containing the training and validation
                    datasets.

            Returns:
                tuple: A tuple containing the English and Portuguese
                    tokenizers. These tokenizers are capable of converting
                    text into subword tokens and back.
        """
        data_train, data_valid = data
        token_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data_train), target_vocab_size=2**15)
        token_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data_train), target_vocab_size=2**15)
        return token_en, token_pt

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper around the `encode` method to allow for efficient
        preprocessing within the data pipeline.

        Args:
            pt (tf.Tensor): The Portuguese sentence.
            en (tf.Tensor): The English sentence.

        Returns:
            tuple: A tuple containing the encoded Portuguese and English
                sentences as TensorFlow tensors.
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

    def filter_by_length(self, pt, en):
        """ filters out sentence pairs where either sentence exceeds the
            maximum length.

        Args:
            pt (tf.Tensor): The Portuguese sentence.
            en (tf.Tensor): The English sentence.

        Returns:
            bool: True if both sentences are within the maximum length,
                otherwise False.
        """
        return tf.size(pt) <= self.max_len and tf.size(en) <= self.max_len

    def filter_wrapper(self, pt, en):
        """ wrapper function for filtering sentences by length

            Args:
                pt (tf.Tensor): The Portuguese sentence.
                en (tf.Tensor): The English sentence.

            Returns:
                tf.Tensor: A boolean tensor indicating whether the sentence
                    pair should be retained.
        """
        return tf.py_function(func=self.filter_by_length,
                              inp=[pt, en], Tout=tf.bool)
