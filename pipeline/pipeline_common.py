# Methods used by several _pipelines
import collections
import random

from config import constants
from config.bdsa_config import _config_
from model import datamodel
from model.bdsa_data import BdsaData
from utils import bdsa_utils, stats_utils
from utils.bdsa_utils import ObservabledDict


class ClusteringOutput:
    def __init__(self):
        self._sa_clusters = ObservabledDict(collections.defaultdict(bdsa_utils.dd_set_generator),
                                            self._on_edit_sa_clusters)
        self._sa_isolated = None
        self._sa_deleted = None

        # Observable dict, so that when it changes, source2pids is reset.
        self._page_clusters = ObservabledDict(collections.defaultdict(bdsa_utils.dd_set_generator),
                                              self._on_edit_page_clusters)
        self._page_isolated = []
        self.att_matches = collections.defaultdict(dict)
        self.page_matches = []
        self.bdsa_data = None
        self._source2pids = None

    def end_algorithm(self):
        """
        Should be called when algorithm has ended: all pages/other elements that where removed for training should be now
        added for evaluation.
        :return:
        """
        self.bdsa_data.end_algorithm()
        # Reset page and SA isolated elements
        self._on_edit_page_clusters()
        self._on_edit_sa_clusters()

    @property
    def page_clusters(self):
        return self._page_clusters

    def set_sa_deleted(self, deleted: set):
        self._sa_deleted = frozenset(deleted)
        self._on_edit_sa_clusters()

    @property
    def sa_deleted(self):
        if self._sa_deleted:
            return self._sa_deleted
        else:
            return []

    @page_clusters.setter
    def page_clusters(self, value):
        self._on_edit_page_clusters()
        self._page_clusters = ObservabledDict(value, self._on_edit_page_clusters)

    def _on_edit_page_clusters(self):
        self._source2pids = None
        self._page_isolated = None

    @property
    def sa_clusters(self):
        return self._sa_clusters

    @sa_clusters.setter
    def sa_clusters(self, value):
        self._on_edit_sa_clusters()
        self._sa_clusters = ObservabledDict(value, self._on_edit_sa_clusters)

    def _on_edit_sa_clusters(self):
        self._sa_isolated = None

    @property
    def source2pids(self):
        """

        :return: inverted source2pids from current page_clusters,
        """
        if not self._source2pids:
            self._source2pids = _get_source2pids(self._page_clusters)
        return self._source2pids

    @property
    def sa_isolated(self):
        if self._sa_isolated is None:
            self._sa_isolated = _find_isolated_nodes(self.bdsa_data.sa2size.keys(), self._sa_clusters, self._sa_deleted)
        return self._sa_isolated

    @property
    def page_isolated(self):
        if not self._page_isolated:
            self._page_isolated = _find_isolated_nodes(self.bdsa_data.page2sa2value.keys(), self.page_clusters, self._sa_deleted)
        return self._page_isolated

    def get_sa2cid(self):
        """
        Get snapshot of inverted sa2cid at the moment of call.
        
        :return: 
        """
        sa2cid = {}
        for cid, source2sas in self.sa_clusters.items():
            for sas in source2sas.values():
                for sa in sas:
                    sa2cid[sa] = cid
        return sa2cid

    def find_name_for_clusters(self) -> dict:
        """
         We give each cluster a name for debugging. This will be the most common attribute name in cluster.
         No need to cache it currently as it is only launched once
        :param output:
        :return:
        >>> output = ClusteringOutput()
        >>> output.sa_clusters = {1: {'s1':\
        [datamodel.source_attribute_factory(None, None, 'nome1'), datamodel.source_attribute_factory(None, None, 'nome2')],\
        's2': [datamodel.source_attribute_factory(None, None, 'nome2')]}, 2: {'s1':[\
        datamodel.source_attribute_factory(None, None, 'nome4')]}}
        >>> output.find_name_for_clusters()
        {1: 'nome2', 2: 'nome4'}
        """

        cid2name = {}
        for cid, source2sas in self.sa_clusters.items():
            sanames = collections.Counter()
            for sas in source2sas.values():
                sanames.update([sa.name for sa in sas if not(sa.is_generated())])
            # Cannot use most_common because it is non-deterministic
            most_common_list = bdsa_utils.most_common_deterministic(sanames, 1)
            cid2name[cid] = most_common_list[0][0] if len(sanames) > 0 else ''
        return cid2name

def _find_isolated_nodes(all_nodes, nid2source2nodes, excluded_nodes):
    """
    Find all isolated nodes (that are not in clusters)
    :param all_nodes:
    :param nid2source2nodes:
    :return:
    """
    isolated_nodes = set(all_nodes)
    for source2nodes in nid2source2nodes.values():
        for nodes in source2nodes.values():
            isolated_nodes.difference_update(nodes)
    if excluded_nodes:
        isolated_nodes.difference_update(excluded_nodes)
    return isolated_nodes

def _get_source2pids(page_clusters:dict):
    """
    :return: inverted source2pids from provided page_clusters.
    """
    source2ids = collections.defaultdict(set)
    for pid, source2pages in page_clusters.items():
        for source in source2pages.keys():
            source2ids[source].add(pid)
    return source2ids

def analyze_clustering_results(clustering_output: ClusteringOutput):
    """
    Return number of isolated-non isolated elements, and size of clusters
    :param clustering_output: 
    :return: 
    """
    stats = {}

    total_attributes = clustering_output.bdsa_data.nb_attributes()
    nb_original_attributes = clustering_output.bdsa_data.nb_original_attributes()
    stats[constants.STATS_NUMBER_ATTRIBUTES] = total_attributes
    stats[constants.STATS_NUMBER_ORIGINAL_ATTRIBUTES] = nb_original_attributes
    # Isolated attributes do not contain generated atts, so %non-isolated also should not consider them.
    nb_non_isolated_original_attributes = nb_original_attributes - len(clustering_output.sa_isolated)
    nb_non_isolated_attributes = total_attributes - len(clustering_output.sa_isolated)
    stats[constants.STATS_PERC_ATTRIBUTES_NON_ISOLATED] = nb_non_isolated_original_attributes / nb_original_attributes
    stats[constants.STATS_NUMBER_NON_ISOLATED_ORIG_ATTRIBUTES] = nb_non_isolated_original_attributes

    stats[constants.STATS_AVG_CLUSTER_SIZE] = stats_utils.safe_divide(nb_non_isolated_attributes,
                                                                      len(clustering_output.sa_clusters.keys()))

    values_for_isolated = sum(clustering_output.bdsa_data.sa2size[sa] for sa in clustering_output.sa_isolated)
    total_values = sum(clustering_output.bdsa_data.sa2size.values())
    stats[constants.STATS_PERC_VALUES_NONISOLATED] = 1 - (float(values_for_isolated) / total_values)
    stats[constants.STATS_NUMBER_PAGES_NON_ISOLATED] = sum(
        len(pages) for source2pages in clustering_output.page_clusters.values() for pages in source2pages.values())
    stats[constants.STATS_NUMBER_PAGE_CLUSTERS] = len(clustering_output.page_clusters.keys())
    stats[constants.STATS_NUMBER_ISOLATED_PAGES] = len(clustering_output.page_isolated)
    return stats


def _add_to_cluster(sa: datamodel.SourceAttribute, cid: int, node2ais: dict,
                    eid2source2element: dict, allow_sa_same_source: bool):
    """
    Add an element still isolated to a cluster, IF sources are different
    :param sa: 
    :param cid: 
    :param node2ais: 
    :return: 
    """
    source2sas = eid2source2element[cid]
    if allow_sa_same_source or not (sa.source in source2sas.keys()):
        source2sas[sa.source].add(sa)
        node2ais[sa] = cid


def _merge_clusters(c1: int, c2: int, node2ais: dict,
                    eid2source2element: dict, allow_sa_same_source: bool):
    """
    Merge 2 clusters, IF there is no source superposition
    :param c1: 
    :param c2: 
    :param node2ais: 
    :return: 
    """
    sources1 = eid2source2element[c1].keys()
    sources2 = eid2source2element[c2].keys()
    if allow_sa_same_source or len(sources1 & sources2) == 0:
        cmin = min(c1, c2)
        cmax = max(c1, c2)
        for source, sas in eid2source2element[cmax].items():
            eid2source2element[cmin][source].update(sas)
            for sa in sas:
                node2ais[sa] = cmin
        del eid2source2element[cmax]


def remove_non_master_attributes(sa_clusters: dict):
    """
    Remove any cluster without a master attribute and whose size is small than a threshold
    :param sa_clusters:
    :return:
    """
    if _config_.get_master_source():
        cid_to_remove = []
        for cid, source2sas in sa_clusters.items():
            # check if minimum attributes or contains at least a master

            if sum(len(sas) for sas in source2sas.values()) < _config_.get_min_size_external_clusters() and \
                    not any(_config_.master_attributes.is_master_attribute(sa)
                        for sas in source2sas.values() for sa in sas):
                cid_to_remove.append(cid)
        for cid in cid_to_remove:
            del sa_clusters[cid]


ITERATIONS_SIMILARITY = 'iterations_similarity'