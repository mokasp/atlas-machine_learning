#!/usr/bin/env python3
from transformers import tform
import tensorflow as tf
import tensorflow_hub as th
import numpy as np
import os


def question_answer(corpus_path):
    """ initiates a question-answer loop where a user can input queries,
        and the system retrieves the most relevant document and provides an
        answer.

        Args:
          corpus_path (str): The path to the directory containing reference
                              documents.

        Process:
          1. Load the Universal Sentence Encoder for semantic search.
          2. Load the BERT model fine-tuned for question-answering.
          3. Load the tokenizer associated with the BERT model.
          4. Collect all reference documents from the specified directory.
          5. Enter a loop where the user can input questions, and the system:
            a. Searches for the most relevant document.
            b. Extracts an answer using BERT.
            c. Displays the answer.
            d. Exits the loop if the user inputs a closing phrase.
      """

    # load the models
    sentence_encoder = th.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/4")
    bert_qa = th.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    tokenizer = tform.BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

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

            # if not, find the document related to the query and retrieve the
            # answer
            references = references_original.copy()
            reference_doc = semantic_search(
                question, references, sentence_encoder)
            answer = get_answer(question, reference_doc, bert_qa, tokenizer)
            print(f'A: {answer}')


def semantic_search(sentence, references, sentence_encoder):
    """ finds the most semantically similar document to the given query using
        the Universal Sentence Encoder.

        Args:
          sentence (str): The query sentence.
          references (list): A list of reference documents.
          sentence_encoder: The Universal Sentence Encoder model.

        Returns:
          str: The most semantically similar document to the query.

        Process:
          1. Encode the query and reference documents using the sentence
             encoder.
          2. Compute the similarity score between the query and each document.
          3. Return the document with the highest similarity score.
    """

    # combine the query with the references and encode this vector
    references = [sentence] + references
    outputs = sentence_encoder(references)['outputs']

    # find similarity score of each document to get most similar doc
    similarity_score = np.inner(outputs, outputs)[0][1:]
    similar_idx = np.argmax(similarity_score) + 1
    similar_doc = references[similar_idx]
    return similar_doc


def get_answer(question, reference, bert_qa, tokenizer):
    """ extracts an answer from the most relevant document using the BERT
        question-answering model.

        Args:
          question (str): The user's question.
          reference (str): The most relevant document.
          bert_qa: The BERT model fine-tuned for question-answering.
          tokenizer: The tokenizer associated with the BERT model.

        Returns:
          str: The answer to the question, or an appropriate message if
               no answer is found.

      Process:
          1. Tokenize the question and reference document.
          2. Create segment embeddings to distinguish between the question
             and document.
          3. Generate attention masks (no padding, so all 1s).
          4. Convert the token IDs, segment embeddings, and attention
             masks to tensors.
          5. Pass these tensors through the BERT model to get start and end
             logits.
          6. If logits are all negative, return a message indicating no
             answer was found.
          7. Otherwise, identify the start and end tokens of the answer and
             return it.
    """

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
    input_word_ids = tf.convert_to_tensor(
        input_token_ids, name='input_word_ids'),
    input_mask = tf.convert_to_tensor(attention_mask, name='input_mask'),
    input_type_ids = tf.convert_to_tensor(
        seg_embedding, name='input_type_ids'),
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
    end = tf.argmax(end_logits) + 2

    # use the indicies to get the answer from the original input tokens and
    # combine them
    answer = ' '.join(input_tokens[start:end])

    return answer
