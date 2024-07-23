#!/usr/bin/env python3

"""_summary_
"""
import requests


def rocket_frequency():
    """_summary_
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"
    launches_response = requests.get(launches_url)
    rockets_response = requests.get(rockets_url)
    launches = launches_response.json()
    rockets_data = rockets_response.json()
    rockets = {rocket["id"]: rocket["name"] for rocket in rockets_data}
    launch_counts = {}

    for launch in launches:
        rocket_id = launch["rocket"]
        rocket_name = rockets.get(rocket_id, "Unknown")
        if rocket_name in launch_counts:
            launch_counts[rocket_name] += 1
        else:
            launch_counts[rocket_name] = 1
    for rocket_name, count in launch_counts.items():
        print(f"{rocket_name}: {count}")


if __name__ == '__main__':
    rocket_frequency()
