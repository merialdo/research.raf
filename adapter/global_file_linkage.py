import json
import collections

from tqdm import tqdm

from utils import string_utils
from config.bdsa_config import _config_
import time


class GlobalLinkageFile:
    """
        This class is used to retrieve the product page URLs associated to each id AND category
        
        ** Note that this class is used only for some scripts, for main schema alignment algorithm
        classes implementing AbstractLinkageAdapter should be used **
        
        The id2cat2urls map is built externally using a record linkage algorithm
        Note that the URLs in the input file are NORMALIZED   
    """

    def __init__(self):
        """
        Retrieves id2cat2urls linkage file name
        """
        self.file = _config_.get_linkage_dexter()
        self.i2c2u_cache = {}

        print ('Opening global linkage JSON file....')
        with open(self.file, 'r') as infile:
            self.i2c2u_cache = json.load(infile, object_pairs_hook=dict_raise_on_duplicates)

        for id, cat2urls in tqdm(self.i2c2u_cache.items(), desc='Initializing global linkage file map...'):
            for cat, urls in cat2urls.items():
                cat2urls[cat] = [string_utils.url_normalizer(url) for url in urls]

    def get_full_map(self):
        """
        Returns the full map of i2u2c, but with normalized URLs instead of original ones.
        Use it ONLY IF NECESSARY as it may be expensive. Otherwise use url_by_id or other methods.
        :return: 
        """
        return self.i2c2u_cache

    def url_by_id(self, id, category):
        """
        Returns the URLs associated with provided ID
        :param id:
        :return:
        """
        if id in self.i2c2u_cache and category in self.i2c2u_cache[id]:
            return self.i2c2u_cache[id][category]
        else:
            return []

    def id_by_url(self, url):
        """
        Returns IDs associated with provided URL (should be one but may be multiple).
        A reverse map is built the first time this method is called

        :param url:
        :return:
        """
        norm_url = string_utils.url_normalizer(url)
        self._build_inverse_map_if_not_exists()
        cat2ids = self.u2c2i[norm_url]
        return [item for sublist in list(cat2ids.values()) for item in sublist]

    def id_by_url_and_category(self, url, category):
        """
        Returns IDs associated with provided URL and category (should be one but may be multiple).
        A reverse map is built the first time this method is called

        :param url:
        :param category:
        :return:
        """
        self._build_inverse_map_if_not_exists()

        norm_url = string_utils.url_normalizer(url)
        return self.u2c2i[norm_url][category]

    def _build_inverse_map_if_not_exists(self):
        if not (hasattr(self, 'u2c2i')):
            self.u2c2i = collections.defaultdict(lambda: collections.defaultdict(list))
            for the_id, cat2urls in self.i2c2u_cache.items():
                for cat, urls in cat2urls.items():
                    for url in urls:
                        self.u2c2i[url][cat].append(the_id)

    def nb_ids(self):
        return len(list(self.i2c2u_cache.keys()))


def dict_raise_on_duplicates(ordered_pairs):
    """Reject duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError("duplicate key: %r" % (k,))
        else:
            d[k] = v
    return d


# Main just loads the linkage file. Useful to check if linkage used is coherent.
if __name__ == '__main__':
    print('Start loading linkage')
    now = time.time()
    test_linkage = GlobalLinkageFile()
    end = time.time()
    print('finished, took %f seconds' % (end - now))
