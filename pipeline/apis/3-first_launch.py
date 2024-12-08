#!/usr/bin/env python3
"""This module fetches and displays the first SpaceX launch."""
import requests
from datetime import datetime
import pytz


def get_first_launch():
    """
    Fetch and display the first SpaceX launch information
    """
    url = "https://api.spacexdata.com/v4/launches"

    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data: {response.status_code}")

    launches = response.json()

    for launch in launches:
        if launch["name"] == "Galaxy 33 (15R) & 34 (12R)":
            first_launch = launch
            break
    else:
        raise RuntimeError("Launch 'Galaxy 33 (15R) & 34 (12R)' not found")

    launch_name = first_launch["name"]
    launch_date_utc = datetime.fromtimestamp(
        first_launch["date_unix"], pytz.utc)
    launch_date_local = launch_date_utc.astimezone(
        pytz.timezone("America/New_York")).isoformat()
    rocket_id = first_launch["rocket"]
    launchpad_id = first_launch["launchpad"]

    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_response = requests.get(rocket_url)
    rocket_name = rocket_response.json()["name"]

    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_response = requests.get(launchpad_url)
    launchpad_data = launchpad_response.json()
    launchpad_name = launchpad_data["name"]
    launchpad_locality = launchpad_data["locality"]

    name_loca = f"{launchpad_name} ({launchpad_locality})"
    print(
        f"{launch_name} ({launch_date_local}) {rocket_name} - {name_loca}"
    )


if __name__ == "__main__":
    get_first_launch()
