#!/usr/bin/env python3

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