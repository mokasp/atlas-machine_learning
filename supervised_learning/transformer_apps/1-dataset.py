#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
  def __init__(self):
    data_train, data_valid = self.load_dataset()
    self.data_train = data_train
    self.data_valid = data_valid
    tokenizer_en, tokenizer_pt = self.tokenize_dataset()
    self.tokenizer_en = tokenizer_en
    self.tokenizer_pt = tokenizer_pt
  
  def load_dataset(self):
    pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
    pt2en_val = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
    return pt2en_train, pt2en_val
  
  def tokenize_dataset(self):
    tokenize_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for _, en in self.data_train), target_vocab_size=2**15)
    tokenize_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, _ in self.data_train), target_vocab_size=2**15)
    return tokenize_en, tokenize_pt

  def encode(self, pt, en):
    pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
        pt.numpy()) + [self.tokenizer_pt.vocab_size+1]
    en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
        en.numpy()) + [self.tokenizer_en.vocab_size+1]
    return pt_tokens, en_tokens