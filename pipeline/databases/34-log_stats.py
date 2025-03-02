#!/usr/bin/env python3
""" tbw tbw """
from pymongo import MongoClient


def log_info(collection):
    """ tbw tbw """
    tracking = {'GET': 0, 'POST': 0, 'PUT': 0, 'PATCH': 0, 'DELETE': 0}
    status = 0
    logs = 0
    for log in collection.find():
        logs += 1
        request = log['method']
        if log['path'] == '/status':
            status += 1
        if request in tracking.keys():
            tracking[request] += 1


    print(str(logs) + ' logs')
    print('Methods:\n\tmethod GET: ' + str(tracking['GET']) +
          '\n\tmethod POST: ' + str(tracking['POST']) +
          '\n\tmethod PUT: ' + str(tracking['PUT']) +
          '\n\tmethod PATCH: ' + str(tracking['PATCH']) +
          '\n\tmethod DELETE: ' + str(tracking['DELETE']))
    print(str(status) + ' status check')


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx
    log_info(collection)