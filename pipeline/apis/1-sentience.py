#!/usr/bin/env python3

"""_summary_

    Returns:
        _type_: _description_
"""
import requests


def getSentientSpecies():
    """_summary_

    Returns:
        _type_: _description_
    """
    url = 'https://swapi-api.hbtn.io/api/species'
    params = {'format': 'json'}
    sentient_homeworlds = set()
    while url:
        data = requests.get(url, params=params).json()
        for species in data['results']:
            if 'classification' in species and 'designation' in species:
                if species['classification'].lower(

                ) == 'sentient' or species['designation'].lower(

                ) == 'sentient':
                    if species['homeworld']:
                        sentient_homeworlds.add(species['homeworld'])
        url = data.get('next')
    return sentient_homeworlds


def sentientPlanets():
    """_summary_

    Returns:
        _type_: _description_
    """
    sentient_homeworlds = getSentientSpecies()
    url = 'https://swapi-api.hbtn.io/api/planets'
    params = {'format': 'json'}
    planets = []
    while url:
        data = requests.get(url, params=params).json()
        for planet in data['results']:
            if planet['url'] in sentient_homeworlds:
                planets.append(planet['name'])
        url = data.get('next')
    return planets
