#!/usr/bin/env python3
""" to be documented, temporary text to check function logic """
import requests


def sentientPlanets():
    """ to be documented, temporary text to check function logic """
    next_page = True
    page = 1
    sentient_worlds = []
    swapi = requests.get('https://swapi-api.hbtn.io/api/species/')
    while next_page:
        if swapi.json()['next'] is None:
            next_page = False
        species = swapi.json()['results']
        for spec in species:
            classification = spec['classification']
            designation = spec['designation']
            if classification == 'sentient' or designation == 'sentient':
                homeworld_url = spec['homeworld']
                if homeworld_url:
                    homeworld_json = requests.get(homeworld_url).json()
                    homeworld = homeworld_json['name']
                    sentient_worlds.append(homeworld)
        page += 1
        url = 'https://swapi-api.hbtn.io/api/species/?page=' + str(page)
        swapi = requests.get(url)
    return sentient_worlds
