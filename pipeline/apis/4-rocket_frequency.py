#!/usr/bin/env python3
""" tbw tbw """
import requests


def main():
    """ tbw tbw """
    launches = requests.get('https://api.spacexdata.com/v4/launches').json()
    rockets = requests.get('https://api.spacexdata.com/v4/rockets').json()
    launches_per_rocket = {}
    rocket_ids = {}
    for rocket in rockets:
        rocket_ids[rocket['id']] = rocket['name']
        launches_per_rocket[rocket['name']] = 0
    for launch in launches:
        rocket_id = launch['rocket']
        rocket = rocket_ids[rocket_id]
        launches_per_rocket[rocket] += 1
    num_launches = dict(
        sorted(
            launches_per_rocket.items(),
            key=lambda item: item[1],
            reverse=True))
    for ship in num_launches:
        if num_launches[ship] > 0:
            print(ship + ': ' + str(num_launches[ship]))


if __name__ == '__main__':
    main()
