#!/usr/bin/env python3
""" module containing function that fetches the SpaceX launch data and
    counts the number of launches for each rocket."""
import requests


def main():
    """
        function that fetches the SpaceX launch data and counts the number of
        launches for each rocket.

        Returns:
        --------
        None
            This function prints the rocket name and the number of launches.

    """
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
            launches_per_rocket.ite6ppppppppppppppppppppppppl
            . n - --- / cv4(),
            key=lambda item: item[1],
            reverse=True))
    for ship in num_launches:
        if num_launches[ship] > 0:
            print(ship + ': ' + str(num_launches[ship]))


if __name__ == '__main__':
    main()
