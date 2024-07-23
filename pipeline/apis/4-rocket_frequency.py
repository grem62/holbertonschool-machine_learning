#!/usr/bin/env python3
"""_summary_"""


import requests

def rocket_frequency():
    """_summary_
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"
    launches = requests.get(launches_url).json()
    rockets = {rocket["id"]: rocket["name"] for rocket in requests.get(rockets_url).json()}
    launch_counts = {}

    for launch in launches:
        rocket_name = rockets.get(launch["rocket"], "Unknown")
        launch_counts[rocket_name] = launch_counts.get(rocket_name, 0) + 1

    for rocket_name, count in launch_counts.items():
        print(f"{rocket_name}: {count}")

if __name__ == '__main__':
    rocket_frequency()
