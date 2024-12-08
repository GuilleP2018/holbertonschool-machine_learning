#!/usr/bin/env python3
"""This module fetches and displays the number of launches per rocket."""
import requests
from collections import defaultdict


def get_launches_per_rocket():
    """
    Fetch and display the number of launches per rocket.
    """
    url = "https://api.spacexdata.com/v4/launches"

    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data: {response.status_code}")

    launches = response.json()

    rocket_launch_count = defaultdict(int)

    for launch in launches:
        rocket_id = launch["rocket"]
        rocket_launch_count[rocket_id] += 1

    rocket_url = "https://api.spacexdata.com/v4/rockets"
    rocket_response = requests.get(rocket_url)
    rockets = rocket_response.json()

    rocket_names = {rocket["id"]: rocket["name"] for rocket in rockets}

    rocket_launch_list = [
        (rocket_names[rocket_id], count)
        for rocket_id, count in rocket_launch_count.items()
    ]

    rocket_launch_list.sort(key=lambda x: (-x[1], x[0]))

    for rocket_name, count in rocket_launch_list:
        print(f"{rocket_name}: {count}")


if __name__ == "__main__":
    get_launches_per_rocket()
