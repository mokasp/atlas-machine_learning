#!/usr/bin/env python3
""" module containing function that returns a list of names of the home
    planets of all sentient species by querying the Swapi API. """
import requests


def sentientPlanets():
    """
        function that returns a list of names of the home planets of all
        sentient species by querying the Swapi API.

        Returns:
        --------
        list
            A list of strings containing the names of the home planets of all
            sentient species. If no sentient species are found, returns an
            empty list.

    """
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
