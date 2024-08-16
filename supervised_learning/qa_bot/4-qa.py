#!/usr/bin/env python3
from transformers import tform
import tensorflow as tf
import tensorflow_hub as th
import numpy as np
import os


def question_answer(corpus_path): 

  # load the models
  sentence_encoder = th.load("https://tfhub.dev/google/universal-sentence-encoder-large/4")
  bert_qa = th.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
  tokenizer = tform.BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
  
  # collect all reference documents
  references_original = []
  for file in os.listdir(corpus_path):
    if not file.startswith('.'):
      with open(corpus_path + '/' + file) as f:
        references_original.append(f.read())

  # begin question-answer loop
  running = True
  while running:
      
      closing = ['exit', 'quit', 'goodbye', 'bye']
      
      # get users query
      question = input('Q: ').lower()
      
      # check if user wants to exit program
      if question in closing:
        print('A: Goodbye')
        running = False
      else:

        # if not, find the document related to the query and retrieve the answer
        references = references_original.copy()
        reference_doc = semantic_search(question, references, sentence_encoder)
        answer = get_answer(question, reference_doc, bert_qa, tokenizer)
        print(f'A: {answer}')

def semantic_search(sentence, references, sentence_encoder):
    
    # combine the query with the references and encode this vector
    references = [sentence] + references
    outputs = sentence_encoder(references)['outputs']
    
    # find similarity score of each document to get most similar doc
    similarity_score = np.inner(outputs, outputs)[0][1:]
    similar_idx = np.argmax(similarity_score) + 1
    similar_doc = references[similar_idx]
    return similar_doc

def get_answer(question, reference, bert_qa, tokenizer):
    
    # tokenize both question and the refence doc
    input_token_ids = tokenizer.encode(question, reference)
    input_tokens = tokenizer.convert_ids_to_tokens(input_token_ids)
    
    # find where the query ends and the document begins
    # and make a vector to indicate their positions
    separator = input_tokens.index('[SEP]')
    seg_embedding = []
    for i in range(len(input_tokens)):
        if i <= separator:
            seg_embedding.append(0)
        else:
            seg_embedding.append(1)

    # make an attention mask as well (there is no padding so it is all 1)
    attention_mask = tf.ones_like(input_token_ids, dtype=tf.int32)

    # convert the vectors to tensors to use as inputs for BERT
    input_word_ids = tf.convert_to_tensor(input_token_ids, name='input_word_ids'),
    input_mask = tf.convert_to_tensor(attention_mask, name='input_mask'),
    input_type_ids = tf.convert_to_tensor(seg_embedding, name='input_type_ids'),
    input_list = [input_word_ids, input_mask, input_type_ids]

    # from the output of bert, extract everything for the start and end logits
    # except for the [CLS] token
    out = bert_qa(input_list)
    start_logits, end_logits = out[0][0][1:], out[1][0][1:]

    # if all logits are less than 0, there is no clear answer
    if np.max(start_logits) < 0 and np.max(end_logits) < 0:
        return 'Sorry, I do not understand your question.'
    
    # find the index of highest logit for both the start and end token
    start = tf.argmax(start_logits) + 1
    end = tf.argmax(end_logits) + 1

    # use the indicies to get the answer from the original input tokens and combine them
    answer = ' '.join(input_tokens[start:end + 1])

    return answer