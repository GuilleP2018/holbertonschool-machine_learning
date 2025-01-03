#!/usr/bin/env python3
"""
Insert a document in a collection
"""


def insert_school(mongo_collection, **kwargs):
    """
    Inserts a new document in a MongoDB collection based on kwargs.
    """
    return mongo_collection.insert_one(kwargs).inserted_id
