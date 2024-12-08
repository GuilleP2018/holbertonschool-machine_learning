#!/usr/bin/env python3
"""This modlue Uses swap api to return a list fo ships that can carry a
given number of passengers"""
import requests


def availableShips(passengerCount):
    """Returns a list of ships that can carry a given number of passengers
    """
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()

        for ship in data.get('results', []):
            try:
                passengers = ship.get('passengers', '0').replace(',', '')
                if passengers.isdigit() and int(passengers) >= passengerCount:
                    ships.append(ship.get('name'))
            except ValueError:
                continue
        url = data.get('next')
    return ships
