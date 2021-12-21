import json
import collections
import os

from adapter.abstract_linkage_adapter import AbstractLinkageAdapter
from model import datamodel
from model.datamodel import SourceSpecifications
from utils import string_utils, io_utils
from config.bdsa_config import _config_
import time

class SourceFileLinkageAdapter(AbstractLinkageAdapter):
    """
        Get IDs of given product pages.
    """

    def __init__(self, linkage_postfix, random_order=False, normalize_url=True, dataset_name=None):
        """
        
        :param linkage_postfix: postfix of json file with linkage [cat]_linkage_[postfix].json. If no postfix is provided,
        then [cat]_linkage.json is used.
        :param random_order if false,  user should ask for all needed URLs of a given source before asking for other sources.
        That's because each time source is changed, new source linkages are loaded and old ones are thrown away.
        """
        self._cached_site = None
        self._cached_category = None
        self._linkage_postfix = linkage_postfix
        self._random_order = random_order
        self._cached_s2u2i = {}
        self.normalize_url = normalize_url
        if dataset_name:
            self.spec_dir = _config_.get_spec_path_from_dataset_name(dataset_name)
        else:
            self.spec_dir = _config_.get_specifications()

    def ids_by_url(self, url, site, category):
        """
        Returns IDs associated with provided URL (should be one but may be multiple).
        A reverse map is built the first time this method is called

        :param category: 
        :param site: 
        :param url:
        :return:
        """
        norm_url = string_utils.url_normalizer(url) if self.normalize_url else url
        source = SourceSpecifications(site, category, None)
        self._build_map_if_not_exists(source)
        this_source_cache = self._cached_s2u2i[source]
        if norm_url in this_source_cache:
            return this_source_cache[norm_url]
        else:
            return []

    def _build_map_if_not_exists(self, source:SourceSpecifications):
        """
        Build a map url --> id for provided site and category.
        :param site: 
        :param category: 
        :return: 
        """
        if source not in self._cached_s2u2i:
            cached_u2i = collections.defaultdict(list)
            filename = '%s_linkage_%s.json' % (source.category, self._linkage_postfix) if self._linkage_postfix else\
                '%s_linkage.json' % source.category
            file_path = os.path.join(self.spec_dir, source.site, filename)
            with open(file_path, 'r') as infile:
                i2u = json.load(infile, object_pairs_hook=dict_raise_on_duplicates)
                for id, urls in i2u.items():
                    for url in urls:
                        norm_url = string_utils.url_normalizer(url) if self.normalize_url else url
                        cached_u2i[norm_url].append(id)
            if not self._random_order:
                self._cached_s2u2i = {}
            self._cached_s2u2i[source] = cached_u2i

    def persist_linkage_data(self, source2pid2urls):
        for source, pid2urls in source2pid2urls.items():
            dir = os.path.join(self.spec_dir, source.site)
            filename = '%s_linkage_%s' % (source.category, self._linkage_postfix) if self._linkage_postfix else \
                '%s_linkage' % source.category
            io_utils.output_json_file({pid: sorted(urls) for pid, urls in pid2urls.items()},
                                      filename, dir, False, True)

def dict_raise_on_duplicates(ordered_pairs):
    """Reject duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError("duplicate key: %r" % (k,))
        else:
            d[k] = v
    return d
