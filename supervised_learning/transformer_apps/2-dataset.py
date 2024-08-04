#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
  def __init__(self):
    data_train, data_valid = self.load_dataset()
    tokenizer_en, tokenizer_pt = self.tokenize_dataset(data_train)
    self.tokenizer_en = tokenizer_en
    self.tokenizer_pt = tokenizer_pt
    self.data_train = data_train.map(self.tf_encode)
    self.data_valid = data_valid.map(self.tf_encode)

  
  def load_dataset(self):
    pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
    pt2en_val = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
    return pt2en_train, pt2en_val
  
  def tokenize_dataset(self, data):

    tokenize_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for _, en in data), target_vocab_size=2**15)
    tokenize_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, _ in data), target_vocab_size=2**15)
    return tokenize_en, tokenize_pt

  def tf_encode(self, pt, en):

    def encode(pt, en):
      pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
          pt.numpy()) + [self.tokenizer_pt.vocab_size+1]
      en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
          en.numpy()) + [self.tokenizer_en.vocab_size+1]
      return pt_tokens, en_tokens

    result_pt, result_en = tf.py_function(lambda pt, en: encode(pt, en), [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en