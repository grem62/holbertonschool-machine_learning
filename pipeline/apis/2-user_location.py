#!/usr/bin/env python3
"""module 2-user_location git Returns the user location
"""
import requests
from datetime import datetime
import sys


def fetch_user_location(api_url):
    """fetches user location"""
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        print(data.get('location', 'Location not set'))

    elif response.status_code == 404:
        print("Not found")

    elif response.status_code == 403:
        reset_time = int(response.headers.get('X-Ratelimit-Reset', 0))
        reset_time = datetime.fromtimestamp(reset_time) - datetime.now()
        minutes = reset_time.total_seconds() // 60
        print("Reset in " + str(int(minutes)) + " min")


if __name__ == '__main__':
    fetch_user_location(sys.argv[1])
