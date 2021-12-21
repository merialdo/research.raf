#coding: utf-8
import hashlib

import unicodedata
import re
import time
import urllib.parse
import os
from math import isfinite

from cachetools import cached, LRUCache

from utils import bdsa_utils

FIND_TOKENS = "(?:[A-Z][a-z]+)|(?:[a-z]+)|(?:[A-Z]+)|(?:[0-9]+[\\.\\,]?[0-9]*)"


def open_tld():
    #http://data.iana.org/TLD/tlds-alpha-by-domain.txt
    file = open(os.path.join(os.path.dirname(__file__),'tlds-alpha-by-domain.txt'), 'r')
    return [a_tld.lower().replace('\n', '') for a_tld in file.readlines() if len(a_tld) <=4]

_tld_ = open_tld()

def url_normalizer(url):
    url_object = urllib.parse.urlparse(url)

    #scheme netloc path params query fragment
    path = url_object.path.strip('/')
    path = re.sub(r"/+", '/', path)
    res = urllib.parse.ParseResult('http', url_object.netloc.replace('www.', ''), path,
                                   url_object.params, url_object.query, '')
    return res.geturl()

@cached(cache=LRUCache(maxsize=1024))
def folding_using_regex(string: object) -> object:
    """
    Normalize strings by:
    * Removing all accents
    * Replacing each sequence of non-alphanumeric character with a single whitespace (foo . ......bar.bre    o --> foo bat bre o)
    * add a space between camelcase words and between a number and a letter (thisIsAWord2this3that --> this is a word 2 this 3 that)
    * lowercase everything
    * trying to put in a single cluster elements looking like a single floating number (3.5 --> single, 3,5 --> single,
        3.5,6.5 --> 3.5 AND 6.5 . Does not work always e.g. does not work for 2,5,10 as a list of numbers
    :param string:
    :return:
    >>> complex_data = '24gr àccèntsAndCamèlCase,commàs,,.dots .dots..numbers2InsideWords.UPPERCASE.lcase .,.,. a1 3.5 '
    >>> folding_using_regex(complex_data)
    '24 gr accents and camel case commas dots dots numbers 2 inside words uppercase lcase a 1 3.5'
    """
    data_intermediate = unicodedata.normalize('NFKD', string).encode('ASCII', 'ignore').decode('ASCII')
    m = re.findall(FIND_TOKENS, data_intermediate)
    folded = ' '.join(m).lower()
    del m
    del data_intermediate
    return folded

def normalize_keyvalues(string:str):
    """
    Normalize attribute name:
    * lowercase
    * trim spaces and ,.;:
    * remove double spaces
    :param string:
    :return:
    >>> normalize_keyvalues("  ,.:   pRoVa   alice bob carlo .  dario ,")
    'prova alice bob carlo . dario'
    >>> normalize_keyvalues("  Additional features: Wi-Fi, flash (built-in): ")
    'additional features: wi-fi, flash (built-in)'
    """
    value = string.lower().strip(" ,.;:")
    res = " ".join(value.split())
    return res

def short_site_name(site):
    """
    Returns a short representation of site name, for instance www.ebay.com --> ebay
    Splits in domains (. separator), starts from the end and return the first non-top level domain element.
    If there are only tld (should not happen) returns first domain.
    :param site:
    :return:
    """
    domains = site.split('.')
    for element in reversed(domains):
        if element not in _tld_:
            return element
    return domains[0]

def weigthed_len(set_of_values):
    """
    Computes a 'weighted length' of the set of possible values of a node.
    This number is bigger as number of values and size (nb of chars) of each value is bigger.
    The reason is that the bigger this number the easier one attribute value may be erroneous
    :param set_of_values:
    :return:
    """
    return sum([min(len(value)/10., 3.) for value in set_of_values])

def timestamp_string_format():
    """
    :return: a string out of current timestamp, usable for filenames 
    """
    return time.strftime("%Y-%m-%d_%H%M%S", time.gmtime())

def is_token_numeric(token:str):
    """
    Return true if token can be converted to numeric
    :param token:
    :return:
    >>> [is_token_numeric('1'), is_token_numeric('1.5'), is_token_numeric('1,5'), is_token_numeric('1,2,3'), is_token_numeric('1a')]
    [True, True, True, False, False]
    """

    return convert_to_numeric(token, -1) is not None

@cached(cache=LRUCache(maxsize=8192))
def convert_to_numeric(value:str, significant_digits=2, return_string=False):
    """
    Convert provided number to numeric.
    :param value:
    :return: the number, or None if number is non-convertible (or Nan-inf)
    """
    try:
        # Allow also comma as floating point separator
        val_norm = value.replace(',','.')
        num = float(val_norm)
        # This is to avoid converting inf or nan to numbers
        if isfinite(num):
            res = bdsa_utils.round_sig(num, significant_digits) \
                if significant_digits > 0 else num
            return str(res) if return_string else res
        else:
            return None
    except ValueError:
        return None

def compute_string_hash(string:str):
    hasher = hashlib.md5(string.encode())
    return hasher.hexdigest()

if __name__ == '__main__':
    u1 = url_normalizer('http://example.com/')
    u2 = url_normalizer('https://www.example.com')
    print("%s %s %s"%(u1,u2,u1==u2))

    print(url_normalizer('www.ebay.com'))

    #print re.match('^($|£|€)[0-9.,]+$', '$25.65')
    print(re.match('^(\$|\€|\£)[0-9.,]+$', '$25.65') is not None)