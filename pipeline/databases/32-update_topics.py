#!/usr/bin/env python3
""" tbw tbw """


def update_topics(mongo_collection, name, topics):
    """ tbw tbw """
    mongo_collection.update_one({'name': name}, {"$set": {'topics': topics}})