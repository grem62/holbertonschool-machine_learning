#!/usr/bin/env python3

"""_summary_
"""
import requests


def sentientPlanets():
    """_summary_
    """
    url = 'https://swapi-api.hbtn.io/api/planets'
    params = {'format': 'json'}
    planets = []
    while url:
        data = requests.get(url, params=params).json()
        for planet in data['results']:
            planets.append(planet['name'])
        url = data.get('next')
    return planets
