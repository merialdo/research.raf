from adapter import mongo_connection_pool
from config.bdsa_config import _config_
from adapter.abstract_linkage_adapter import AbstractLinkageAdapter
from pymongo import MongoClient


class MongoLinkageAdapter(AbstractLinkageAdapter):

    def __init__(self):
        client = mongo_connection_pool.get_mongo_connection(_config_.get_mongo_host())
        self.db = client[_config_.get_mongo_db()]

    def ids_by_url(self, url, site, category):
        results = list(self.db.Products.find({'url': url, 'category': category, 'website': site}))
        if len(results) == 0:
            raise Exception("No product with provided URL")
        elif len(results) > 1:
            raise Exception("More than 1 product with provided URL")
        else:
            return results[0]['ids']


