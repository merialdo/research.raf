from pymongo import MongoClient
from tqdm import tqdm

from config.bdsa_config import _config_
from model.datamodel import SourceSpecifications

from adapter import abstract_specifications_adapter, mongo_connection_pool


class MongoSpecificationsGenerator(abstract_specifications_adapter.AbstractSpecificationsGenerator):
    def __init__(self):
        client = mongo_connection_pool.get_mongo_connection(_config_.get_mongo_host())
        self.db = client[_config_.get_mongo_db()]

    def source_names_ordered_linkage_decreasing(self):
        return []

    def nb_of_specs(self):
        return self.db.Schemas.count()

    def _specifications_generator_intern(self, normalize_data=True):
        category_chosen = _config_.get_category()

        all_sources = list(self.db.Schemas.find({}, {'category': 1, 'website': 1}))
        for rsource in tqdm(all_sources, desc='Retrieve all source...'):
            site = rsource['website']
            category = rsource['category']
            if not category_chosen or category_chosen == category:
                source_spec = list(
                    self.db.Products.find({"category": category, 'website': site}, {'_id': 0, 'url': 1, 'spec': 1}))
                source_spec_adapted = abstract_specifications_adapter \
                    .build_specifications_object(source_spec, normalize_data, normalize_data, False)

                yield SourceSpecifications(site, category, source_spec_adapted)

    def persist_specifications(self, sites2category2page2att2value):
        """
        Not implemented for the moment
        :param sites2category2page2att2value: 
        :return: 
        """
        raise NotImplementedError()

    def _source_specifications_intern(self, site, category):
        source_spec = list(
            self.db.Products.find({"category": category, 'website': site}, {'_id': 0, 'url': 1, 'spec': 1}))
        source_spec_adapted = abstract_specifications_adapter \
            .build_specifications_object(source_spec, False, False, True)

        return SourceSpecifications(site, category, source_spec_adapted)
