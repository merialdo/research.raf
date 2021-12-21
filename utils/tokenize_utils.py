## Utility methods to convert value into set of tokens
from nltk import word_tokenize
from cachetools import cached, LRUCache, TTLCache


def value2token_set(value):
    return frozenset(value2token(value))

@cached(cache=LRUCache(maxsize=8192))
def value2token(value):
    return word_tokenize(value)
