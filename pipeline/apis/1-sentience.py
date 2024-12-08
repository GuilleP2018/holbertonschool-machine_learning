#!/usr/bin/env python3
"""This module uses the SWAPI API to return a list of
homeworld planets of all sentient species."""
import requests


def sentientPlanets():
    """
    Fetch a list of homeworld planets of all sentient species.
    Maintains the order provided by the SWAPI API.
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets_list = []

    while url:
        response = requests.get(url)

        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch data: {response.status_code}")

        data = response.json()

        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()
            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get("homeworld")
                if homeworld_url:
                    homeworld_response = requests.get(homeworld_url)
                    if homeworld_response.status_code == 200:
                        homeworld_name = homeworld_response.json().get("name")
                        if homeworld_name:
                            planets_list.append(homeworld_name)
        url = data.get("next")

    return planets_list
