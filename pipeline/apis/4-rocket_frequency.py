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

    # Sort launch_counts by the number of launches in descending order
    sorted_launch_counts = sorted(launch_counts.items(), key=lambda item: item[1], reverse=True)

    for rocket_name, count in sorted_launch_counts:
        print("{}: {}".format(rocket_name, count))

if __name__ == '__main__':
    rocket_frequency()

