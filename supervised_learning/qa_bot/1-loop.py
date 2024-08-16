#!/usr/bin/env python3

running = True
while running:
    closing = ['exit', 'quit', 'goodbye', 'bye']
    question = input('Q: ').lower()
    if question in closing:
        print('A: Goodbye')
        running = False
    else:
        print('A: ')