#!/usr/bin/env python3
""" to be documented, temporary text to check function logic """
import requests
import sys

def main():
    """ to be documented, temporary text to check function logic """
    github = sys.argv[1]
    user_json = requests.get(github).json()
    if user_json['status'] == '404':
        print('Not found')
    elif user_json['status'] == '403':
        x = user_json['X-Ratelimit-Reset']
        print('Reset in ' + str(x) + ' min')
    else:
        print(user_json['location'])


if __name__ == '__main__':
    main()
