import collections
import itertools

import random
## These are technical workarounds for pickling default dicts
from math import floor, log10

import git
from cachetools import cached, LRUCache


def dd_dict_generator():
    return collections.defaultdict(dict)

def dd_set_generator():
    return collections.defaultdict(set)

def dd_list_generator():
    return collections.defaultdict(list)

def dd2_set_generator():
    return collections.defaultdict(dd_set_generator)

def counter_generator():
    return collections.Counter()

def dd_int_generator():
    return collections.defaultdict(int)

def dd_counter_generator():
    return collections.defaultdict(counter_generator)

def dd_float_generator():
    return collections.defaultdict(float)

def dd2_int_generator():
    return collections.defaultdict(dd_int_generator)


def list_padder(res:list, size=5, pad_element=''):
    """
    If provided list has less than 'size' elements, pads it with provided pad text
    :param res:
    :return:
    """
    res.extend([(pad_element, 0)] * (size - len(res)))
    return res

def multidict_inverter(a2b:dict, a_filter=None, b_filter=None, a_group_filter=None):
    """
    Invert dict of set
    :param a2b:
    :param a_filter: potential filter for A elements
    :param b_filter: potential filter for B elements
    :return:
    """
    b2a = collections.defaultdict(set)
    for a, bs in a2b.items():
        if a_filter is None or a_filter(a):
            for b in bs:
                if b_filter is None or b_filter(b):
                    b2a[b].add(a)

    if a_group_filter:
        res = {}
        for b, _as in b2a.items():
            if a_group_filter(_as):
                res[b] = _as
    else:
        res = dict(b2a)
    return res

def build_dict(data, key_getter, value_getter, multi=True, key_getter_returns_multi=False, value_getter_returns_multi=False):
    """
    Build a multiple dict, given a collection of elements

    :param data: the collection
    :param key_getter: the way to get the key(s) from each element in data
    :param value_getter: the way to get the value(s) from each element in data
    :param multi: Whether the dictionary should be mutivalues (set of values) or single-values
    :param key_getter_returns_multi: If key_getter return multiple keys, then we will create (or update) an entry with each of these keys
    :param value_getter_returns_multi: If value_gette return multiple values, then we will add of them to the entry (Makes no sense if multi=False)
    :return:
    """
    res = collections.defaultdict(set) if multi else {}
    for element in data:
        key_s = key_getter(element)
        value_s = value_getter(element)
        for key in key_s if key_getter_returns_multi else [key_s]:
            for value in value_s if value_getter_returns_multi else [value_s]:
                if multi:
                    res[key].add(value)
                else:
                    res[key] = value
    return res

def randomly_remove_elements(elements:list, keep_probability:float, random_function=random.random):
    """
    Randomly remove elements from a list
    :param elements: list of elements
    :param keep_probability: for each element, this is the probability that this element should be kept
    :return: list
    >>> fake_random_elements = (x/5 for x in range(5))
    >>> fake_random = lambda : next(fake_random_elements)
    >>> elements = [10,1,8,2,3]
    >>> randomly_remove_elements(elements, 0.3, fake_random)
    [8, 2, 3]
    >>> fake_random_elements = (x/5 for x in range(5))
    >>> randomly_remove_elements(elements, 0.6, fake_random)
    [3]
    """
    res = []
    for element in elements:
        if random_function() > keep_probability:
            res.append(element)
    return res

def random_true_false(ratio:float):
    """
    Return true with a probability of RATIO, otherwise false
    :param ratio:
    :return:
    """
    return random.random() < ratio

def most_common_deterministic(counter:collections.Counter, k:int):
    """
    Return most common K elements of a collections, if 2 elements are equal return first in order defined by values.
    :param counter:
    :param k:
    :return:
    """
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:k]

def dict_printer(adict: dict, dict_of_dict=True):
    """
    Simple dict printer
    :param adict:
    :return:
    """
    return '\n'.join(str(key) +' --- ' + (str(sorted(value.items())) if dict_of_dict else str(value)) for key, value in sorted(adict.items()))

class ObservabledDict:
    """
    Classical dictionary, but when it changes it calls a provided method
    """

    def __init__(self, wrapped_dict:dict, callback):
        self.wrapped_dict = wrapped_dict
        self.call = callback

    def __setitem__(self, key, value):
        self.wrapped_dict[key] = value
        self.call()

    def __getitem__(self, item):
        return self.wrapped_dict[item]

    def __delitem__(self, key):
        del self.wrapped_dict[key]
        self.call()

    def clear(self):
        self.wrapped_dict.clear()
        self.call()

    def items(self):
        return self.wrapped_dict.items()

    def values(self):
        return self.wrapped_dict.values()

    def keys(self):
        return self.wrapped_dict.keys()

    def __len__(self):
        return len(self.wrapped_dict)

def round_sig(x, sig=2):
    """
    Round a number to 2 significant digits
    :param sig:
    :return:
    """
    if x == 0:
        return 0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def split_list(data, splitter, return_list=False):
    """
    Split a list into pieces according to a splitter
    :param data:
    :param splitter:
    :param return_list:
    :return:
    """
    res = collections.defaultdict(list if return_list else set)
    for element in data:
        element_dict = res[splitter(element)]
        element_dict.append(element) if return_list else element_dict.add(element)
    return dict(res)

def nb_pairs_in_cluster(nb_elements):
    """
    Return number of pairs in a cluster
    >>> nb_pairs_in_cluster(4)
    6
    >>> nb_pairs_in_cluster(1)
    0
    """
    return int(nb_elements * (nb_elements - 1) / 2)

def getbool(text:str):
    return text.lower() in ['true', 'yes']

def partition_data(elements, func, reverse=False):
    sorted_data = sorted(elements, key=func, reverse=reverse)
    return {a:list(b) for a,b in itertools.groupby(sorted_data, key=func)}

def find_contained(id2set:dict):
    """
    Given a set of sets (identified), find all sets that are completely contained in another set
    :param sets:
    :return:
    >>> find_contained({'a':{1,2,3}, 'b':{3,2}, 'c': {1,5},'d':{1}, 'e':{1,5}})
    ['b', 'c', 'd']
    >>> elems = {x: set(range(x)) for x in range(10)}
    >>> find_contained(elems)
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    """
    all_elements = set()
    all_setids = id2set.keys()
    all_setids_sorted = sorted(all_setids, key=lambda x: len(id2set[x]))
    candidate_pairs = set(tuple(x) for x in itertools.combinations(all_setids_sorted, 2))
    element2setids = collections.defaultdict(set)
    for pid, aset in id2set.items():
        all_elements.update(aset)
        for elem in aset:
            element2setids[elem].add(pid)
    all_grouped_setids = set(frozenset(s) for s in element2setids.values())
    # For a given element, a present set id cannot be a subset of a missing set id
    for setids in all_grouped_setids:
        missing_setids = all_setids - setids
        pairs_to_remove = set(tuple(x) for x in itertools.
                              product(setids, missing_setids))
        candidate_pairs.difference_update(pairs_to_remove)
    return sorted(set(x[0] for x in candidate_pairs))


def get_git_commit():
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:10], repo.head.commit.summary
    except git.InvalidGitRepositoryError:
        return "not git repo", "not git repo"


def get_git_commit_message(commit_id):
    try:
        repo = git.Repo(search_parent_directories=True)
        message = repo.commit(commit_id).message #head.object.hexsha[:10], repo.head.commit.summary
        message = message.replace('\n','').replace('\r','')
    except Exception:
        message = None
    if not message:
        message = 'Message not available'
    return message