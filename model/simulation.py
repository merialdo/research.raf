"""
This class contains all simulations made to alterate the original content of the dataset,
in order to make specific experiments.
"""
import collections
import random

import itertools
from typing import List
import copy

from adapter.abstract_specifications_adapter import AbstractSpecificationsGenerator
from model.clusters import FreeClusterRules
from model.datamodel import SourceSpecifications, Page
from pipeline import cluster_utils
from pipeline.cluster_utils import WeightedEdge
from config.bdsa_config import _config_, SourceRemovalRule, LinkageRemovalRule
from utils import bdsa_utils

SuspendedPage = collections.namedtuple('SuspendedPage', 'page pids key2value')


class Simulation:

    def __init__(self, sgen:AbstractSpecificationsGenerator):
        self.remove_random_linkage = _config_.get_linkage_removal_rule() == LinkageRemovalRule.RANDOM
        # If we must delete sources according to linkage, then we must get the sources ordered by linkage
        if _config_.get_source_removal_rule() != SourceRemovalRule.NONE:
            list_sources_sorted = sgen.source_names_ordered_linkage_decreasing()
            sources_mandatory = [source for source in list_sources_sorted if source.site in _config_.get_sources_mandatory()]
            other_sources = [source for source in list_sources_sorted if source.site not in _config_.get_sources_mandatory()]
            nb_othersource_to_keep = _config_.get_number_sources_kept() - len(sources_mandatory)
            if nb_othersource_to_keep > 0:
                if _config_.get_source_removal_rule() == SourceRemovalRule.KEEP_MORE_LINKAGE:
                    self.sources_to_keep = sources_mandatory + other_sources[:nb_othersource_to_keep]
                elif _config_.get_source_removal_rule() == SourceRemovalRule.KEEP_LESS_LINKAGE:
                    self.sources_to_keep = sources_mandatory + other_sources[-nb_othersource_to_keep:]
                else:
                    self.sources_to_keep = sources_mandatory + random.sample(other_sources, k= nb_othersource_to_keep)
            else:
                self.sources_to_keep = sources_mandatory
        else:
            self.sources_to_keep = None
        self._suspended_pages:List[SuspendedPage] = []
        self._suspended_linkages = {}
        self._old_linkage = None

    def do_delayed_operations(self, bdsa_data):
        """
        Execute all operations that were delayed (ignored for training, applied for other)
        :return:
        """
        for sus_page in self._suspended_pages:
            bdsa_data.add_page(sus_page.page, sus_page.key2value, sus_page.pids)
        for pid, source2pages in self._suspended_linkages.items():
             bdsa_data.pid2source2pages[pid] = source2pages
        if _config_.get_ratio_linkage_added() > 0:
            bdsa_data.pid2source2pages = self._old_linkage

    def potentially_remove_linkage_clusters(self, pid2source2pages):
         """
         Remove linkage if asked in configuration
         :param pid2source2pages:
         :return:
         """
         self._suspended_linkages = _remove_cluster(
             pid2source2pages, _config_.get_linkage_removal_rule(), _config_.get_percentage_linkage_kept())

    def check_and_add_product_page(self, bdsa_data, page:Page, key2values:dict, pids):
        """
        Check whether page can be added or not, according to simulation configuration.
        Currently the only case is when do_delete_pages_without_linkage is activated, and page
        is not associated with any product ID.
        If true, add it immediately. Otherwise, put the operation in "delay" and add them
        :param pids:
        :param product_page_adder:
        :return:
        """
        if not _config_.do_delete_pages_without_linkage() or len(pids) > 0:
            bdsa_data.add_page(page, key2values, pids)
        else:
            self._suspended_pages.append(SuspendedPage(page, pids, key2values))

    def do_add_source(self, source):
        """
        Check if a source has to be added. According to configuration, one may remove random sources, or sources
        with most or less linkage
        :param source:
        :return:
        """
        if _config_.get_source_removal_rule() == SourceRemovalRule.NONE:
            return True

        return source in self.sources_to_keep

    def add_wrong_linkages(self, bdsa_data):

        percent_added = _config_.get_ratio_linkage_added()
        if percent_added > 0:
            # Save old linkages so we can retrieve them later
            self._old_linkage = copy.deepcopy(bdsa_data.pid2source2pages)

            # This is current nb of pairs of specifications in linkage
            current_pairs = sum(sum(len(pgs[0]) * len(pgs[1]) for pgs in itertools.combinations(source2pages.values(), 2))
                                for source2pages in bdsa_data.pid2source2pages.values())

            new_expected_pairs = current_pairs * percent_added
            new_pairs = 0
            nb_iterations = 0
            pids = list(bdsa_data.pid2source2pages.keys())
            pages = list(bdsa_data.page2sa2value.keys())

            # Add new IDs to pages until we reach enough linkages OR too many iterations
            while new_pairs < new_expected_pairs and nb_iterations < len(bdsa_data.page2sa2value.keys()) * 5:
                pid = random.choice(pids)
                page = random.choice(pages)
                source2pages_pid = bdsa_data.pid2source2pages[pid]
                if page in source2pages_pid[page.source]:
                    continue
                new_pairs += sum(len(pages) for source, pages in source2pages_pid.items() if source != page.source)
                source2pages_pid[page.source].add(page)




def generate_random_pairs(max):
    """
    Generate random pair of number without repetitions
    :param max: 
    :return: 
    """
    already_seen = set()
    size = 0
    while size < max * (max - 1) / 2:
        res = tuple(sorted([random.randint(0, max), random.randint(0, max)]))
        if res[0] != res[1] and res not in already_seen:
            size += 1
            already_seen.add(res)
            yield res

        # def all_pairs(pids, pages, choice_function):
#     for i in pids:
#         for j in pages:
#             if choice_function(i,j):
#                 yield (i,j)
#
# def choose_pairs(pids, pages, sample):
#     total_pairs = len(pids) * len(pages)
#     random.sample()

def _remove_cluster(linkage_clusters: dict, removal_rule: LinkageRemovalRule, ratio_linkage_keep: float):
    """
    Remove linkage cluster to simulate different conditions of input
    :return:
    >>> linkage_clusters_orig = {10: {1: [1,2,3], 2: [2,3]}, 20: {1:[1,2], 2: [3]}, 30: {1: [1,2,3,4]}}
    >>> linkage_clusters_copy = linkage_clusters_orig.copy()
    >>> _remove_cluster(linkage_clusters_copy, LinkageRemovalRule.KEEP_SMALL_CLUSTERS, 0.7)
    {10: {1: [1, 2, 3], 2: [2, 3]}}
    >>> linkage_clusters_copy
    {20: {1: [1, 2], 2: [3]}, 30: {1: [1, 2, 3, 4]}}
    >>> linkage_clusters_copy = linkage_clusters_orig.copy()
    >>> _remove_cluster(linkage_clusters_copy,  LinkageRemovalRule.KEEP_BIG_CLUSTERS, 0.7)
    {20: {1: [1, 2], 2: [3]}}
    >>> linkage_clusters_copy
    {10: {1: [1, 2, 3], 2: [2, 3]}, 30: {1: [1, 2, 3, 4]}}
    >>> linkage_clusters_copy = linkage_clusters_orig.copy()
    >>> _remove_cluster(linkage_clusters_copy, LinkageRemovalRule.KEEP_SMALL_CLUSTERS, 0.4)
    {10: {1: [1, 2, 3], 2: [2, 3]}, 30: {1: [1, 2, 3, 4]}}
    >>> linkage_clusters_copy
    {20: {1: [1, 2], 2: [3]}}
    >>> linkage_clusters_copy = linkage_clusters_orig.copy()
    >>> _remove_cluster(linkage_clusters_copy, LinkageRemovalRule.KEEP_BIG_CLUSTERS, 0.4)
    {20: {1: [1, 2], 2: [3]}, 30: {1: [1, 2, 3, 4]}}
    >>> linkage_clusters_copy
    {10: {1: [1, 2, 3], 2: [2, 3]}}
    >>> linkage_clusters_copy = linkage_clusters_orig.copy()
    >>> sorted(_remove_cluster(linkage_clusters_copy, LinkageRemovalRule.RANDOM, 0.1).items())
    [(10, {1: [1, 2, 3], 2: [2, 3]}), (20, {1: [1, 2], 2: [3]}), (30, {1: [1, 2, 3, 4]})]
    >>> linkage_clusters_copy
    {}
    >>> linkage_clusters_copy = linkage_clusters_orig.copy()
    >>> ignore = _remove_cluster(linkage_clusters_copy, LinkageRemovalRule.RANDOM, 0.4)
    >>> len(linkage_clusters_copy)
    1
    """
    total_linked = sum(len(pages) for source2pages in linkage_clusters.values() for pages in source2pages.values())
    elements_to_remove = round(total_linked * (1 - ratio_linkage_keep))

    removed = {}
    nb_removed = 0
    pids_to_remove = []
    if removal_rule in {LinkageRemovalRule.RANDOM, LinkageRemovalRule.KEEP_SMALL_CLUSTERS, LinkageRemovalRule.KEEP_BIG_CLUSTERS}:
        if removal_rule == LinkageRemovalRule.RANDOM:
            pids_sorted = list(linkage_clusters.keys())
            random.shuffle(pids_sorted)
        elif removal_rule in {LinkageRemovalRule.KEEP_SMALL_CLUSTERS, LinkageRemovalRule.KEEP_BIG_CLUSTERS}:
            remove_biggest_first = removal_rule == LinkageRemovalRule.KEEP_SMALL_CLUSTERS
            # https://docs.python.org/2/library/heapq.html
            # [... nlargest] perform best for smaller values of n. For larger values, it is more efficient to use the sorted()
            # so we use sorted as tipically N is same order of magnitude then the whole dataset
            pids_sorted = sorted(linkage_clusters.keys(), reverse=remove_biggest_first,
                                    key=lambda pid: sum(len(pages) for pages in linkage_clusters[pid].values()))
        for pid in pids_sorted:
            pids_to_remove.append(pid)
            nb_removed += sum(len(pages) for pages in linkage_clusters[pid].values())
            if nb_removed >= elements_to_remove:
                break

    for pid in pids_to_remove:
        removed[pid] = linkage_clusters.pop(pid)
    return removed