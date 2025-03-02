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
    print('Methods:')
    print('        method GET: ' + str(tracking['GET']))
    print('        method POST: ' + str(tracking['POST']))
    print('        method PUT: ' + str(tracking['PUT']))
    print('        method PATCH: ' + str(tracking['PATCH']))
    print('        method DELETE: ' + str(tracking['DELETE']))
    print(str(status) + ' status check')


client = MongoClient('mongodb://127.0.0.1:27017')
collection = client.logs.nginx
log_info(collection)