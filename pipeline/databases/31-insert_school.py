#!/usr/bin/env python3
""" tbw tbw """


def insert_school(mongo_collection, **kwargs):
    """ tbw tbw """
    new_doc = {}
    for key, value in kwargs.items():
        new_doc[key] = value
    new_school = mongo_collection.insert_one(new_doc)
    return new_school.inserted_id