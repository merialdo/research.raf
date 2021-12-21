from abc import ABC, abstractmethod

from config.bdsa_config import _config_
from model.datamodel import SourceSpecifications
from utils import string_utils

SPEC_KEY = 'spec'
URL_KEY = 'url'

# Generator for specifications import.
# Note that URLs are always normalized, unless a specific parameter is provided


class AbstractSpecificationsGenerator(ABC):

    @abstractmethod
    def nb_of_specs(self):
        pass

    def specifications_generator(self, normalize_data=True):
        """As specifications_generator but retrieves directory from config."""
        #TODO: after refacto I think it could be deleted

        for element in self._specifications_generator_intern(normalize_data):
            yield element

    @abstractmethod
    def source_names_ordered_linkage_decreasing(self):
        """
        Get a sorted list of source names, ordered by linkage decreasing.
        This list may be hard-coded in input, as its calculation is costy.
        TODO Further developments may add a system to "cache" this information
        The algorithm is heuristic, as there is not a clear definition of "source with most linkage"
        :return:
        """
        pass

    @abstractmethod
    def _specifications_generator_intern(self, normalize_data=True):
        pass

    def source_specifications(self, site, category) -> SourceSpecifications:
        """
        As source_specifications but retrieves directory from config
        :param site: 
        :param category: 
        :return: 
        """

        return self._source_specifications_intern(site, category)

    @abstractmethod
    def _source_specifications_intern(self, site, category):
        pass

    @abstractmethod
    def persist_specifications(self, sites2category2page2att2value, linkage_retriever=None):
        pass

    @abstractmethod
    def persist_specifications_functional(self, output_tag, source_linkage_retriever, source_spec_retriever):
        """
        Persist specifications using external functions
        :param source_linkage_retriever: generator method for linkage data for each source
        :param source_spec_retriever: generator method for spec data for each source
        :return:
        """
        pass


def build_specifications_object(source_data, normalize_key, normalize_values, normalize_url):
    output = {}
    for page in source_data:
        if normalize_url:
            page[URL_KEY] = string_utils.url_normalizer(page[URL_KEY])
        specs = page[SPEC_KEY]
        if normalize_key or normalize_values:
            for key in list(specs.keys()):
                value = specs[key]
                if normalize_values:
                    value = string_utils.normalize_keyvalues(value)
                if normalize_key:
                    del specs[key]
                    key = string_utils.normalize_keyvalues(key)
                specs[key] = value
        url = page[URL_KEY]
        if url in output:
            print("*** WARNING: duplicated url, merging: %s" % url)
            output[url].update(page[SPEC_KEY])
        else:
            output[url] = page[SPEC_KEY]
    return output
