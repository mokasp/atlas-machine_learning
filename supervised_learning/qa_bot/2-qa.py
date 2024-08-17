#!/usr/bin/env python3
""" this script provides functionality for answering questions based on a
    reference  text using a pre-trained BERT model. It utilizes TensorFlow
    Hub to load the BERT  model and the Hugging Face Transformers library for
    tokenization. Users can  interactively ask questions and receive answers
    derived from the provided text.

    Dependencies:
        - tensorflow: library for numerical computation and machine learning.
        - tensorflow_hub: library for loading pre-trained models from TF Hub.
        - transformers: A library by Hugging Face for Transformer-based models
                        and tokenization.
        - numpy: A library for numerical operations in Python.

    Functions:
        - answer_loop(reference): Starts an interactive question-answering
            loop where users can ask questions about the reference text.
        - question_answer(question, reference, bert_qa, tokenizer): Processes
            the question and reference text to provide the most likely answer
            using the BERT model.

    Usage:
        1. Call `answer_loop(reference)` with the reference text to begin the
           interactive session.
        2. Users can input questions and receive answers until they type a
           termination command such as 'exit', 'quit', 'goodbye', or 'bye'.
        3. The `question_answer(question, reference, bert_qa, tokenizer)`
           function is used internally to handle the process of question
           answering.

"""
import tensorflow as tf
import tensorflow_hub as th
import transformers as tform
import numpy as np


def answer_loop(reference):
    """ initiates a loop for interactive question answering. Users can input
            questions and receive answers based on the provided reference
            text. The loopcontinues until a termination command is issued.

        Args:
            reference (str): The reference text used to answer questions.

    """
    # load in models
    bert_qa = th.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    tokenizer = tform.BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

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
            answer = question_answer(question, reference, bert_qa, tokenizer)
            print(f'A: {answer}')


def question_answer(question, reference, bert_qa, tokenizer):
    """ provides an answer to a question based on the reference text using
        a BERT model. The function tokenizes the input, prepares the data for
        the model, and extracts the answer from the model's output.

        Args:
            question (str): The question to be answered.
            reference (str): The reference text containing the information
                needed to answer the question.
            bert_qa (tf.Module): The BERT model for question answering.
            tokenizer (transformers.BertTokenizer): The tokenizer used to
                process the input text.

        Returns:
            str: The answer to the question extracted from the reference
                text. If no clear answer is found, a default message is
                returned.
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
