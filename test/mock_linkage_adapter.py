from adapter import mongo_connection_pool
from config.bdsa_config import _config_
from adapter.abstract_linkage_adapter import AbstractLinkageAdapter


class MockLinkageAdapter(AbstractLinkageAdapter):

    def __init__(self):
        print ("initialise mock linkage")
        pass

    def ids_by_url(self, url, site, category):
        return []

