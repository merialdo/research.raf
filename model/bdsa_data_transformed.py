import collections

from cachetools import LRUCache, cached
from tqdm import tqdm

from model import datamodel
from model.datamodel import SourceAttribute
from utils import bdsa_utils, tokenize_utils, string_utils
from config.bdsa_config import _config_
from utils.bdsa_utils import round_sig

MIN_TOKENS_FOR_COMMON_EXTRACTION = 3


def _extract_common_tokens(page2sa2value:dict):
    """
    Extract tokens that are present in EVERY occurrence of an attribute
    :param page2sa2value:
    :return:
    """
    sa2tokens2occs = collections.defaultdict(bdsa_utils.counter_generator)
    sa2size = collections.defaultdict(int)
    for page, sa2value in page2sa2value.items():
        for sa, value in sa2value.items():
            token_set = tokenize_utils.value2token_set(value)
            # Check for frequent tokens
            sa2tokens2occs[sa].update(token_set)
            sa2size[sa] += 1
    sa2common_tokens = {}
    for sa, token2occs in sa2tokens2occs.items():
        if sa2size[sa] >= MIN_TOKENS_FOR_COMMON_EXTRACTION:
            sa2common_tokens[sa] = set(token for token, occs in token2occs.items() if
                                            occs / sa2size[sa] >= _config_.get_common_token_ratio())
        else:
            sa2common_tokens[sa] = set()

    return sa2common_tokens


class BdsaDataTransformed:
    """
    Data from BDSA transformed.
    In order to avoid keeping maps of original and transformed value maps, which is error-prone, this object should be rebuild
    after each modification in data
    """
    def __init__(self, page2sa2value:dict):
        # This map is converted immediately and not lazily, as some values after conversion might merge
        # Es: {'en, fr': 5, 'fr, en': 2} --> {{en,fr} : 7}
        self._sa2transformed_values2occs = collections.defaultdict(bdsa_utils.counter_generator)

        self.sa2common_tokens = {}

        # Values present in more than 1 source
        self._non_isolated_values = set()

        # Nb of distinct attribute a value is present in
        self._value2nb_distinct_attributes = set()

        self._nb_total_atts = None

        # This one is lazily converted (to avoid too much memory usage).
        self._original_page2sa2value = page2sa2value

        self._common_tokens_rule = _config_.get_common_token_ratio() != 0
        self._common_token_ratio = _config_.get_common_token_ratio()

        self._initialize_maps(page2sa2value)

    def _initialize_maps(self, page2sa2value):

        # The page2sa2value is passed 2 time, one for computing common tokens (if common token rule applies)
        # and one to build maps and make all transformations. This adds a bit of overload (not that much as tokenizer is
        # cached) but after a lot of pain I realized it is WAAAAAAAAAAAAAAAAY easier to read
        if self._common_tokens_rule:
            self.sa2common_tokens = _extract_common_tokens(page2sa2value)

        value2first_source = {}
        value2sas = collections.defaultdict(set)

        # Build all maps here
        for page, sa2value in tqdm(page2sa2value.items(), desc='Building transformed data...'):
            for sa, value in sa2value.items():
                transformed_value = self.transform_value(sa, value)
                self._sa2transformed_values2occs[sa][transformed_value] += 1
                value2sas[transformed_value].add(sa)
                self._check_isolated(sa, transformed_value, value2first_source)

        del value2first_source
        self._value2nb_distinct_attributes = {value: len(sas) for value, sas in value2sas.items()}
        del value2sas

    def _check_isolated(self, sa, transformed_value, value2first_source):
        """
        Add value to non-isolated list if it is
        :param sa:
        :param transformed_value:
        :param value2first_source:
        :return:
        >>> sa1 = datamodel.source_attribute_factory('dummy', 's1', 'a1')
        >>> sa12 = datamodel.source_attribute_factory('dummy', 's1', 'a2')
        >>> sa2 = datamodel.source_attribute_factory('dummy', 's2', 'a2')
        >>> sa3 = datamodel.source_attribute_factory('dummy', 's3', 'a3')
        >>> value2first_source = {}
        >>> dt = BdsaDataTransformed({})
        >>> dt._check_isolated(sa1, 1, value2first_source)
        >>> dt._check_isolated(sa1, 2, value2first_source)
        >>> dt._check_isolated(sa1, 2, value2first_source)
        >>> dt._check_isolated(sa1, 3, value2first_source)
        >>> dt._check_isolated(sa12, 3, value2first_source)
        >>> dt._check_isolated(sa12, 3, value2first_source)
        >>> dt._check_isolated(sa12, 4, value2first_source)
        >>> dt._check_isolated(sa2, 4, value2first_source)
        >>> dt._check_isolated(sa1, 5, value2first_source)
        >>> dt._check_isolated(sa2, 5, value2first_source)
        >>> dt._check_isolated(sa3, 5, value2first_source)
        >>> dt._non_isolated_values
        {4, 5}
        """
        # If this value isn't already in non-isolated values, then we have to check if it is non-isolated
        if transformed_value not in self._non_isolated_values:
            # If this value was already found elsewhere in another source, then it is non-isolated.
            if transformed_value in value2first_source and value2first_source[transformed_value] != sa.source:
                self._non_isolated_values.add(transformed_value)
                # Remove it from value2first_source as it is now useless
                del value2first_source[transformed_value]
            else:
                value2first_source[transformed_value] = sa.source

    def get_transformed_value2occs(self, sa:SourceAttribute):
        return self._sa2transformed_values2occs[sa]

    def is_value_non_isolated(self, value, transformed=True, sa=None):
        value_mod = value if transformed else self.transform_value(sa, value)
        return value_mod in self._non_isolated_values

    def is_common_value(self, value):
        if self._nb_total_atts is None:
            self._nb_total_atts = len(self._sa2transformed_values2occs)
        nb_atts = self.nb_distinct_occurrences_attributes(value)
        return nb_atts > 15 and nb_atts / self._nb_total_atts > _config_.get_max_frequency_single_value()

    def nb_distinct_occurrences_attributes(self, value):
        return self._value2nb_distinct_attributes[value]

    @cached(cache=LRUCache(maxsize=8192))
    def get_sa2value_for_page(self, page):
        res = {}
        for sa, value in self._original_page2sa2value[page].items():
            res[sa] = self.transform_value(sa, value)
        return res

    def transform_value(self, sa:SourceAttribute, value):
        """
        Transform value into set of tokens, and remove most frequent tokens
        :param sa:
        :param value:
        :return:
        """
        new_token_set = _transform_single_value(value)

        if self._common_tokens_rule:
            new_token_set = self._remove_common_tokens(sa, new_token_set)
        return frozenset(new_token_set)

    def _remove_common_tokens(self, sa:SourceAttribute, token_set:set):
        """
        Remove common tokens from set
        :param sa:
        :param token_set:
        :return:
        """

        res = token_set - self.sa2common_tokens[sa]
        if len(res) == 0:
            ## This is to avoid removing all elements of a token
            return token_set
        else:
            return res


@cached(cache=LRUCache(maxsize=32768))
def _transform_single_value(value):
    token_set = tokenize_utils.value2token_set(value)
    # Approximation
    new_token_set = set()
    for val in token_set:
        numeric = string_utils.convert_to_numeric(val, return_string=True)
        if numeric:
            new_token_set.add(numeric)
        else:
            new_token_set.add(val)
    return frozenset(new_token_set)