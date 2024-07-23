#!/usr/bin/env python3

import requests


def availableShips(passengerCount):
    url = 'https://swapi-api.hbtn.io/api/starships'
    params = {'format': 'json'}
    ships = []

    while url:
        r = requests.get(url, params=params)
        data = r.json()
        for ship in data['results']:
            try:
                if int(ship['passengers']) >= passengerCount:
                    ships.append(ship['name'])
            except (ValueError, KeyError):
                continue
        url = data.get('next')

    return ships
