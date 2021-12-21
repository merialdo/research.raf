from pymongo import MongoClient

mongo_pool = {}

def get_mongo_connection(host_and_port):
    """
    Returns a connection to mongo, avoiding 
    :param host_and_port: 
    :return: 
    """
    if host_and_port not in mongo_pool:
        mongo_pool[host_and_port] = MongoClient('mongodb://%s/'%(host_and_port))
    return mongo_pool[host_and_port]