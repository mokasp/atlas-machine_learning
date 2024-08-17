#!/usr/bin/env python3
""" this script initiates an interactive question-answering loop. Users can
    input questions and receive answers. The loop continues until the user
    decides to exit the program by typing a termination command.

    Usage:
        1. Run the script to start the interactive loop.
        2. Enter questions to receive responses.
        3. To exit the loop, type one of the termination commands such as
           'exit', 'quit', 'goodbye', or 'bye'.

    Functionality:
        - Handles user input and provides a response based on the input.
        - Continues to prompt the user for input until a termination command
          is given.

    Termination Commands:
        - 'exit'
        - 'quit'
        - 'goodbye'
        - 'bye'

    Example:
        Q: What is the capital of France?
        A:

        Q: exit
        A: Goodbye
"""

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
        # if not, return blank answer
        print(f'A: ')
