from adapter.abstract_linkage_adapter import AbstractLinkageAdapter
from adapter.abstract_specifications_adapter import AbstractSpecificationsGenerator
from config.bdsa_config import _config_

from adapter.file_specifications_adapter import FileSpecificationsGenerator
from adapter.mongo_linkage_adapter import MongoLinkageAdapter
from adapter.mongo_specifications_adapter import MongoSpecificationsGenerator
from adapter.source_file_linkage_adapter import SourceFileLinkageAdapter


def linkage_factory(linkage_file_suffix: str = None, random_order=False, normalize_url=True,
                    dataset_name=None) -> AbstractLinkageAdapter:
    """
    Factory of a linkage adapter
    :return: 
    """
    spec_source = _config_.get_specifications_source()

    if spec_source == _config_.SpecificationsSource.MONGO:
        linkage_adapter = MongoLinkageAdapter()
    elif spec_source == _config_.SpecificationsSource.FILE:
        linkage_adapter = SourceFileLinkageAdapter(linkage_file_suffix, random_order, normalize_url, dataset_name)
    else:
        raise Exception("Unknown specifications source")
    return linkage_adapter

def spec_factory(dataset_name=None, cat=None) -> AbstractSpecificationsGenerator:
    """
    Factory of specification adapter
    :return: 
    """
    spec_source = _config_.get_specifications_source()

    if spec_source == _config_.SpecificationsSource.MONGO:
        specifications_adapter = MongoSpecificationsGenerator()
    elif spec_source == _config_.SpecificationsSource.FILE:
        specifications_adapter = FileSpecificationsGenerator(dataset_name, cat)
    else:
        raise Exception("Unknown specifications source")
    return specifications_adapter