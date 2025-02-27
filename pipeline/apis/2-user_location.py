#!/usr/bin/env python3
""" module containing function that etches and prints the location of a
    GitHub user from the provided API URL. """
import requests
import sys
import datetime


def main():
    """
        function that etches and prints the location of a GitHub user from
        the provided API URL.

        Parameters:
        -----------
        api_url : str
            The full API URL for the GitHub user
            (e.g., https://api.github.com/users/username).

        Returns:
        --------
        None
            This function prints the location of the user or appropriate
            error messages.

    """
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
        print(user_json.json()['location'])


if __name__ == '__main__':
    main()
