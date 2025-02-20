#!/usr/bin/env python3
""" to be documented, temporary text to check function logic """
import requests
import sys
import datetime

def main():
    """ to be documented, temporary text to check function logic """
    github = sys.argv[1]
    user_json = requests.get(github)
    if user_json.status_code == 404:
        print('Not found')
    elif user_json.status_code == 403:
        time = user_json.headers['X-Ratelimit-Reset']
        time = datetime.datetime.fromtimestamp(int(time))
        cur_time = datetime.datetime.now()
        minutes = round((time - cur_time).total_seconds() / 60.0)
        print('Reset in ' + str(minutes) + ' min')
    else:
        print(user_json['location'])


if __name__ == '__main__':
    main()
