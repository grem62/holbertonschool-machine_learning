#!/usr/bin/env python3

"""_summary_

    Returns:
        _type_: _description_
"""
import requests


def availableShips(passengerCount):
    """_summary_

    Args:
        passengerCount (_type_): _description_

    Returns:
        _type_: _description_
    """
    url = 'https://swapi-api.hbtn.io/api/starships'
    params = {'format': 'json'}
    ships = []

    while url:
        r = requests.get(url, params=params)
        data = r.json()
        for ship in data['results']:
            try:
                if int(ship['passengers'].replace(',', '')) >= passengerCount:
                    ships.append(ship['name'])
            except (ValueError, KeyError):
                continue
        url = data.get('next')

    return ships
