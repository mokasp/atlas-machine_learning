#!/usr/bin/env python3
import requests


def availableShips(passengerCount):
    next_page = True
    page = 1
    filtered_ships = []
    swapi =  requests.get('https://swapi-api.hbtn.io/api/starships/')
    while next_page:
        if swapi.json()['next'] == None:
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

