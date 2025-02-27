#!/usr/bin/env python3
""" module containing function that returns a list of ships that can hold a
    given number of passengers by querying the Swapi API. """
import requests


def availableShips(passengerCount):
    """
        function that returns a list of ships that can hold a given number
        of passengers by querying the Swapi API.

        Parameters:
        -----------
        passengerCount : int
            The minimum number of passengers that the ship should be able to
            hold. Ships with a higher or equal capacity will be included in
            the returned list.

        Returns:
        --------
        list
            A list of dictionaries, where each dictionary contains the 'name'
            and 'model' of a ship that can hold at least the specified
            number of passengers. If no ships meet the criteria, an empty list
            is returned.

    """
    next_page = True
    page = 1
    filtered_ships = []
    swapi = requests.get('https://swapi-api.hbtn.io/api/starships/')
    while next_page:
        if swapi.json()['next'] is None:
            next_page = False
        else:
            ship_subset = swapi.json()['results']
            for ship in ship_subset:
                num_passengers = ship['passengers']
                try:
                    num_passengers = num_passengers.replace(",", "")
                    num_passengers = int(num_passengers)
                except ValueError:
                    num_passengers = 0
                if num_passengers >= passengerCount:
                    filtered_ships.append(ship['name'])
            page += 1
            url = 'https://swapi-api.hbtn.io/api/starships/?page=' + str(page)
            swapi = requests.get(url)
    return filtered_ships
