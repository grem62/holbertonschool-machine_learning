#!/usr/bin/env python3
"""_summary_
"""

import requests


def first_launch():
    """_summary_
    """
    launche = 'https://api.spacexdata.com/v4/launches'
    rocket = 'https://api.spacexdata.com/v4/rockets'
    launchpad = 'https://api.spacexdata.com/v4/launchpads'
    response = requests.get(launche)
    launches = response.json()
    response = requests.get(rocket)
    rockets = response.json()
    response = requests.get(launchpad)
    launchpads = response.json()

    sorted_launches = sorted(launches, key=lambda x: x['date_local'])

    latest_launch = sorted_launches[0]

    rocket_id = latest_launch['rocket']
    launchpad_id = latest_launch['launchpad']
    for rocket in rockets:
        if rocket['id'] == rocket_id:
            rocket_name = rocket['name']
    for launchpad in launchpads:
        if launchpad['id'] == launchpad_id:
            launchpad_name = launchpad['name']
            launchpad_locality = launchpad['locality']

    launch_name = latest_launch['name']
    launch_date = latest_launch['date_local']

    print("{} ({}) {} - {} ({})".format(
        launch_name, launch_date, rocket_name,
        launchpad_name, launchpad_locality))


if __name__ == '__main__':
    first_launch()
