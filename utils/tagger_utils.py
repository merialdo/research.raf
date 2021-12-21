import collections
from typing import Callable, Any

from cachetools import cached, LRUCache
from nltk import word_tokenize, ngrams

from utils import tokenize_utils

DEFAULT_MAX_NGRAM = 10
DEFAULT_MIN_NGRAM = 1


class Tagger:
    """
    Class that will tag provided text with correspondent cluster names
    """
    def __init__(self, val2element: dict, extractor: Callable[[list, Any], Any], joiner: Callable[[Any, Any, Any], Any],
                 tag_every_combo:bool, min_ngram=DEFAULT_MIN_NGRAM, max_ngram=DEFAULT_MAX_NGRAM):
        """
        Replace val2cid with val2cluster_name (i.e. most frequent attribute name in cluster)
        :param val2cid:
        :param aname2cid:
        :param cid2most_frequent_name:
        """
        self.val2element = {tuple(tokenize_utils.value2token(val)): element for val, element in val2element.items()}
        self.extractor = extractor
        self.joiner = joiner
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self._tag_every_combination = tag_every_combo

    # Is there a python library that does this? Currently I only found POS taggers (like NLTK) and not agnostic taggers

    @cached(cache=LRUCache(maxsize=8192))
    def tag(self, value: str, excluded_clusters:frozenset):
        """
        Method that actually tags a text
        :param value:
        :return:
        """
        tokenize = tokenize_utils.value2token(value)
        if self._tag_every_combination:
            return self._tag_tokens_full(tokenize, excluded_clusters)
        else:
            return self._tag_tokens(tokenize, excluded_clusters)

    def _tag_tokens(self, tokens: list, excluded_clusters):
        """
        Browse through n-grams of the word to detect snippets to tag, then calls recursively on left and right part

        :param tokens:
        :param first_iteration:
        :return:
        """
        max_relative_size = len(tokens)
        for n in range(min(self.max_ngram, max_relative_size), self.min_ngram - 1, -1):
            for position, gram in enumerate(ngrams(tokens, n)):
                if gram in self.val2element and self.val2element[gram] != excluded_clusters:
                    #Recursive step
                    left_part = self._tag_tokens(tokens[0:position], excluded_clusters)
                    right_part = self._tag_tokens(tokens[position + n:],excluded_clusters)
                    tag = self.extractor(gram, self.val2element[gram] - excluded_clusters)
                    return self.joiner(left_part, tag, right_part)
        return self.joiner(None, self.extractor(tokens, None), None)

    def _tag_tokens_full(self, tokens: list, excluded_clusters):
        """
        Browse through n-grams of the word to detect snippets to tag.
        Tags ALL possible combinations found.

        :param tokens:
        :param first_iteration:
        :return:
        """
        max_relative_size = len(tokens)
        res = collections.defaultdict(list)
        for n in range(min(self.max_ngram, max_relative_size), self.min_ngram - 1, -1):
            for position, gram in enumerate(ngrams(tokens, n)):
                for cname in self.val2element.get(gram, set()) - excluded_clusters:
                    res[cname].append(' '.join(gram))
        return res

#
# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
