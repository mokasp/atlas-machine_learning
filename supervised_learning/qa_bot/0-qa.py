#!/usr/bin/env python3
""" this script defines a `question_answer` function that utilizes a
    pre-trained BERT model for answering questions based on a reference
    document. The function takes a question and a reference document as input
    and returns the most likely answer from the document.

    Dependencies:
        - tensorflow: A library for numerical computation using data flow
            graphs.
        - tensorflow_hub: A library for reusable machine learning modules.
        - transformers: A library for state-of-the-art Natural Language
            Processing (NLP) models.
        - numpy: A library for numerical operations in Python.

    Function:
        - question_answer(question, reference):
            Takes a question and a reference document, processes them using
            BERT for question answering, and returns the extracted answer.

        Arguments:
            - question (str): The question to be answered.
            - reference (str): The document containing the information needed
                to answer the question.

        Returns:
            - str: The extracted answer from the reference document. If no
                clear answer is found, it returns a default message indicating
                that the question is not understood.

        Steps:
            1. Load the BERT model and tokenizer from TF Hub and Hugging Face.
            2. Tokenize the input question and reference document.
            3. Create segment embeddings and attention masks required for BERT.
            4. Convert tokenized inputs into TensorFlow tensors.
            5. Run the BERT model to obtain start and end logits.
            6. Determine start and end indices of the answer from the logits.
            7. Extract and return the answer from the reference document.

        Example:
            question = "What is the capital of France?"
            reference = "The capital of France is Paris."
            answer = question_answer(question, reference)
            print(answer)  # Output: "Paris"
"""
import tensorflow as tf
import tensorflow_hub as th
import transformers as tform
import numpy as np


def question_answer(question, reference):
    """ uses a pre-trained BERT model to answer a question based on a
        reference document.

        This function loads the BERT model and tokenizer, tokenizes the input
        question and reference document, and processes them to extract the
        most likely answer from the document. The answer is determined based
        on the start and end logits output by BERT.

        Args:
            question (str): The question to be answered.
            reference (str): The document containing the information needed to
                answer the question.

        Returns:
            str: The extracted answer from the reference document. If no clear
                answer is found, it returns a default message indicating that
                the question is not understood.
    """
    # load in models
    bert_qa = th.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    tokenizer = tform.BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

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
    end = tf.argmax(end_logits) + 2

    # use the indicies to get the answer from the original input tokens and combine them
    answer = ' '.join(input_tokens[start:end])

    return answer