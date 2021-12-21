import collections
from abc import ABC, abstractmethod
from model.datamodel import SourceAttribute, Page
from config.bdsa_config import _config_

class AbstractCluster(ABC):
    @abstractmethod
    def add_element_if_allowed(self, node) -> bool:
        """
        Add an element to a cluster if it is allowed, returns whether it was added
        :param node:
        :return:
        """
        pass

    @abstractmethod
    def merge_cluster(self, c2):
        """
        Add elements
        :param c2:
        :return:
        """
        pass

    @abstractmethod
    def nodes(self) -> set:
        """
        Return list of nodes
        :return:
        """
        pass


class AbstractClusterRules(ABC):
    @abstractmethod
    def is_pair_mergeable(self, node1, node2) -> bool:
        """
        Tells if 2 elements can be merged in a cluster
        :param node1:
        :param node2:
        :return:
        """
        pass

    def build_cluster(self, node1, node2) -> AbstractCluster:
        res = self._build_empty_cluster()
        #Should be always allowed
        res.add_element_if_allowed(node1)
        res.add_element_if_allowed(node2)
        return res

    @abstractmethod
    def _build_empty_cluster(self) -> AbstractCluster:
        pass

# For source attributes


class SaCluster(AbstractCluster):
    """
    A cluster of attributes. This class provides method to add a single attribute or merge with another one,
    and verify if all rules are complied:
    * If Master attribute parameters are activated, there should not be 2 master attributes in cluster
    * More generally, 2 attributes in the cluster should never appear in the same page. It is possible to allow
      generated attributes to skip this rule
    """

    def __init__(self, sa2urls: dict):
        self.sa2urls = sa2urls
        self.urls = set()
        self.sas = set()

    def add_element_if_allowed(self, node:SourceAttribute):
        """
        Add element if it is allowed
        :param node:
        :return:
        """
        page_overlapping = len(self.sa2urls[node] & self.urls) > 0
        page_overlap_exclude_clause = _config_.exclude_same_page_for_generated() and node.is_generated()
        multiple_masters = _config_.master_attributes.is_master_attribute(node) and \
            any(_config_.master_attributes.is_master_attribute(sa) for sa in self.sas)
        if (not page_overlapping or page_overlap_exclude_clause) and not multiple_masters:
            self.sas.add(node)
            if not page_overlap_exclude_clause:
                self.urls.update(self.sa2urls[node])
            return True
        return False

    def merge_cluster(self, other):
        atts_from_same_pages = len(self.urls & other.urls) > 0
        # The new attribute is a master, and there is already a master in cluster
        multiple_masters = any(_config_.master_attributes.is_master_attribute(sa) for sa in other.sas) and \
            any(_config_.master_attributes.is_master_attribute(sa) for sa in self.sas)
        if not atts_from_same_pages and not multiple_masters:
            self.sas.update(other.sas)
            self.urls.update(other.urls)
            return True
        return False

    def nodes(self):
        return self.sas


class SaClusterRules(AbstractClusterRules):
    """
    Clustering rule for attributes.
    * If Master attribute parameters are activated, there should not be 2 master attributes in cluster
    * More generally, 2 attributes in the cluster should never appear in the same page. It is possible to allow
      generated attributes to skip this rule
    """
    def __init__(self, sa2urls:dict):
        self.sa2urls = sa2urls

    def is_pair_mergeable(self, node1: SourceAttribute, node2: SourceAttribute) -> bool:
        attributes_from_same_page = len(self.sa2urls[node1] & self.sa2urls[node2]) > 0
        same_page_exclusion_clause_activated = _config_.exclude_same_page_for_generated() \
                                               and (node1.is_generated() or node2.is_generated())
        both_masters = _config_.master_attributes.is_master_attribute(node1) and \
            _config_.master_attributes.is_master_attribute(node2)
        return (same_page_exclusion_clause_activated or not attributes_from_same_page) and not both_masters

    def _build_empty_cluster(self,) -> SaCluster:
        return SaCluster(self.sa2urls)

# For pages


class InterSourceCluster(AbstractCluster):
    """
    See InterSourceClusterRules
    """
    def __init__(self):
        self.sources = set()
        self.node_list = set()

    def add_element_if_allowed(self, node):
        if node.source not in self.sources:
            self.sources.add(node.source)
            self.node_list.add(node)
            return True
        return False

    def merge_cluster(self, other: SaCluster):
        if len(self.sources & other.sources) == 0:
            self.node_list.update(other.node_list)
            self.sources.update(other.sources)
            return True
        return False

    def nodes(self):
        return self.node_list


class InterSourceClusterRules(AbstractClusterRules):
    """
    Cluster rules applicable to pages and sources, in which we deny elements from same source in same cluster.

    """
    def __init__(self):
        pass

    def is_pair_mergeable(self, node1, node2) -> bool:
        return node1.source != node2.source

    def _build_empty_cluster(self,) -> InterSourceCluster:
        return InterSourceCluster()

class FreeCluster(AbstractCluster):
    """
    See FreeClusterRules
    """
    def __init__(self):
        self.node_list = set()

    def add_element_if_allowed(self, node):
        self.node_list.add(node)
        return True

    def merge_cluster(self, other: SaCluster):
        for att in other.nodes():
            self.add_element_if_allowed(att)
        return True

    def nodes(self):
        return self.node_list


class FreeClusterRules(AbstractClusterRules):
    """
    Cluster rules applicable to pages and sources, in which we allow clustering under any clause.

    """
    def __init__(self):
        pass

    def is_pair_mergeable(self, node1, node2) -> bool:
        return True

    def _build_empty_cluster(self,) -> FreeCluster:
        return FreeCluster()
