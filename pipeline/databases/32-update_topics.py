#!/usr/bin/env python3
"""
Changes all topics of a school document
"""


def update_topics(mongo_collection, name, topics):
    """
    Changes all topics of a school document based on the given name.
    """
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
