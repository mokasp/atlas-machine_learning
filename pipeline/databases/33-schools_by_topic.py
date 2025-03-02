#!/usr/bin/env python3
""" tbw tbw """


def schools_by_topic(mongo_collection, topic):
    """ tbw tbw """
    filtered_schools = []
    for school in mongo_collection.find():
        try:
            if topic in school['topics']:
                filtered_schools.append(school)
        except:
            pass
    return filtered_schools