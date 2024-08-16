#!/usr/bin/env python3
from transformers import tform
import tensorflow as tf
import tensorflow_hub as th
import numpy as np
import os


def question_answer(corpus_path):
  sentence_encoder = th.load("https://tfhub.dev/google/universal-sentence-encoder-large/4")
  bert_qa = th.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
  tokenizer = tform.BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
  references_original = []
  for file in os.listdir(corpus_path):
    if not file.startswith('.'):
      with open(corpus_path + '/' + file) as f:
        references_original.append(f.read())

  running = True
  while running:
      closing = ['exit', 'quit', 'goodbye', 'bye']
      question = input('Q: ').lower()
      if question in closing:
        print('A: Goodbye')
        running = False
      else:
        references = references_original.copy()
        reference_doc = semantic_search(question, references, sentence_encoder)
        answer = get_answer(question, reference_doc, bert_qa, tokenizer)
        print(f'A: {answer}')

def semantic_search(sentence, references, sentence_encoder):
    references = [sentence] + references
  
    outputs = sentence_encoder(references)['outputs']
    similarity_score = np.inner(outputs, outputs)[0][1:]
    similar_idx = np.argmax(similarity_score) + 1
    similar_doc = references[similar_idx]
    return similar_doc

def get_answer(question, reference, bert_qa, tokenizer):
    input_token_ids = tokenizer.encode(question, reference)
    input_tokens = tokenizer.convert_ids_to_tokens(input_token_ids)
    separator = input_tokens.index('[SEP]')

    seg_embedding = []
    for i in range(len(input_tokens)):
        if i <= separator:
            seg_embedding.append(0)
        else:
            seg_embedding.append(1)
    attention_mask = tf.ones_like(input_token_ids, dtype=tf.int32)

    input_word_ids = tf.convert_to_tensor(input_token_ids, name='input_word_ids'),
    input_mask = tf.convert_to_tensor(attention_mask, name='input_mask'),
    input_type_ids = tf.convert_to_tensor(seg_embedding, name='input_type_ids'),

    input_list = [input_word_ids, input_mask, input_type_ids]

    out = bert_qa(input_list)
    start_logits, end_logits = out[0][0][1:], out[1][0][1:]
    if np.max(start_logits) < 0 and np.max(end_logits) < 0:
        return None
    start = tf.argmax(start_logits) + 1
    end = tf.argmax(end_logits) + 1

    answer = ' '.join(input_tokens[start:end + 1])

    return answer