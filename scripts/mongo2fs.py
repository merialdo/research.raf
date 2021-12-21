from pymongo import MongoClient
import os
import collections

from adapter import adapter_factory, mongo_connection_pool
from config.bdsa_config import _config_
from utils import io_utils, string_utils

WEBSITE = 'website'


def export2fs():
    """
    Export synthetic dataset generated in mongo, to file system
    :return:
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client.SyntheticDataset
    all_sources = db.Schemas.find({}, {'category': 1, 'website': 1})

    output_dir = os.path.join(_config_.get_output_dir(),
                              'synthetic_dataset' + string_utils.timestamp_string_format())
    os.makedirs(output_dir)
    for rsource in all_sources:
        site = rsource[WEBSITE]
        category = rsource['category']

        # Build file
        sitedir = os.path.join(output_dir, site)
        if not os.path.exists(sitedir):
            os.makedirs(sitedir)
        source_spec = list(db.Products.find({"category": category, 'website': site}, {'_id': 0, 'url': 1, 'spec': 1}))
        io_utils.output_json_file(source_spec, '%s/%s_spec' % (sitedir, category), timestamp=False)
            # Now build linkage
        linkage = collections.defaultdict(list)
        for page in source_spec:
            pid = page['url'].split('/')[1]
            linkage[pid].append(page['url'])
        io_utils.output_json_file(linkage,'%s/%s_linkage' % (sitedir, category), timestamp=False)

def fs2mongo():
    """
    Convert BDSA dataset in file system to mongo.
    :return:
    """
    # Mongo dataset
    client = mongo_connection_pool.get_mongo_connection(_config_.get_mongo_host())
    db = client[_config_.get_mongo_db()]

    linkage_adapter = adapter_factory.linkage_factory()
    linkages = collections.defaultdict(set)
    for source in adapter_factory.spec_factory().specifications_generator():
        descriptors = []
        attributes = set()
        for url, spec in source.pages.items():
            spec_modified = {key.replace('.', '_dot_').replace('$', '_dollar_'): value for key, value in spec.items() if key not in _config_.get_excluded_attribute_names()}
            ids = linkage_adapter.ids_by_url(url, source.site, source.category)
            for pid in ids:
                linkages[pid].add(url)
            descriptors.append({'url': url, 'category': source.category, 'website': source.site,
                                'ids':ids, 'spec':spec_modified, 'linkage':[]})
            attributes.update(spec_modified.keys())
        db.Products.insert_many(descriptors)
        db.Schemas.insert_one({WEBSITE: source.site, 'category': source.category, 'attributes':list(attributes)})
    # fix 'linkages'
    for pid, urls in linkages.items():
        db.Products.update_many({'url': {'$in':list(urls)}}, {'$set': {'linkage': list(urls)}})




if __name__ == '__main__':
    fs2mongo()