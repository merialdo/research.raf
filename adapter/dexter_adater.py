import json
import requests
from numpy import random

__author__ = 'fpiai'

# This code is most copied by occurrence-extractor ##
# This is done to avoid importing LOTS of OE and knowledge web modules (as empirically seen)
# and to avoid some import problems ##

# TODO However this is a workaround so a more stable solution should be defined ##
# In particular, dexter core and dexter utils should be separated in 2 project or 2 modules
# (see how this is possible with Python) ##


HOSTS = [
    {"host": "localhost", "port": 9200},
    # {"host": "alfred:hosDFsGyBagVati%5BLYtCgV38@54.246.25.95", "port": 80},
    # {"host": "alfred:hosDFsGyBagVati%5BLYtCgV38@54.220.194.254", "port": 80},
]


def search_document_urls(keyword, max_results=250):
    payload = {'query': {'match': {'text': keyword}},
               'fields': []}

    content = json.dumps(payload, sort_keys=True, indent=4)
    headers = {'Content-Type': 'application/json'}

    request_url = build_random_url_request() + "_search?size=" + str(max_results)

    res = requests.post(
        request_url,
        data=content,
        headers=headers,
        timeout=60
    )

    res = json.loads(res.content)

    urls = []
    for hit in res['hits']['hits']:
        if "http" in hit['_id']:
            urls += [hit['_id']]
    return urls


def build_random_url_request():
    return build_request_url(get_host())


def get_host():
    return random.choice(HOSTS)


def build_request_url(host):
    return "http://" + host["host"] + ":" + str(host["port"]) + "/"
