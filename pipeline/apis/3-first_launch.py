#!/usr/bin/env python3
""" module containing function that fetches the first SpaceX launch from the
    API and displays its details"""
import requests


def main():
    """
        function that fetches the first SpaceX launch from the API and
        displays its details.

        Returns:
        --------
        None
            Prints the formatted information about the first launch.

    """
    req = requests.get('https://api.spacexdata.com/v4/launches/')
    launches = req.json()
    for launch in launches:
        launch_name = launch['name']
        if launch_name == 'Galaxy 33 (15R) & 34 (12R)':
            launch_date = launch['date_local']
            rocket_id = launch['rocket']
            rockets = requests.get(
                'https://api.spacexdata.com/v4/rockets').json()
            for rocket in rockets:
                if rocket['id'] == rocket_id:
                    rocket_name = rocket['name']
            launchpad = launch['launchpad']
            launchpads = requests.get(
                'https://api.spacexdata.com/v4/launchpads').json()
            for pad in launchpads:
                if pad['id'] == launchpad:
                    launchpad_name = pad['name']
                    launchpad_locality = pad['locality']
            print(launch_name + ' (' + launch_date + ') ' + rocket_name +
                  ' - ' + launchpad_name + ' (' + launchpad_locality + ')')

            return


if __name__ == '__main__':
    main()
