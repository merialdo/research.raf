import collections
import itertools
import random

from model.clusters import AbstractClusterRules
from utils import stats_utils, bdsa_utils


def aggregate_clusters(edges: list, eid2source2element: dict, score:int, cluster_rules: AbstractClusterRules,
                                  compare_previous_limit=None):
    """
    Aggregate former cluster (eid2source2element) with new edges
    :param edges:
    :param eid2source2element:
    :param cluster_rules:
    :param compare_previous_limit:
    :return:
    """
    for source2elements in eid2source2element.values():
        for source_pair in itertools.combinations(source2elements.keys(), 2):
            for element_pair in itertools.product(source2elements[source_pair[0]], source2elements[source_pair[1]]):
                edges.append(WeightedEdge(element_pair[0], element_pair[1], score))
    partition_using_agglomerative(edges, eid2source2element, cluster_rules, compare_previous_limit)


def partition_using_agglomerative(edges: list, eid2source2element: dict, cluster_rules: AbstractClusterRules,
                                  compare_previous_limit=None):
    """
    Cluster graph in this way:
    - Initially each node is a cluster
    - Foreach edge (sorted by decreasing weight):
     - If nodes for this edge are in different clusters, if a specific precondition for merging is met (tipically there
        should be elements from same source)
     - Stop at MIN_EDGE_WEIGHT
    Inspired (but different) by paper 'Clustering large probabilistic graph'
    :param edges: list of edges with nodes and weight. Each node must have an attribute 'source'.
    :param eid2source2element: the dictionary that represents the output
    :param cluster_rules: class that says if elements can be merged, and build cluster objects.
    :param: compare_previous_limit: Compute detailed similarity (costly) if only if gross similarity is bigger \
    than this. If None, never compute detailed similarity. Cf stats_utils.compare_clusters
    :return:
    """

    edges_sorted = sorted(edges, key=lambda e: (e.weight, e.node1, e.node2), reverse=True)
    node2cid = {}
    cid2cluster = {}
    cluster_counter = 0
    for edge in edges_sorted:
        node1 = edge.node1
        node2 = edge.node2
        c1 = node2cid[node1] if node1 in node2cid else None
        c2 = node2cid[node2] if node2 in node2cid else None
        if c1 is None and c2 is None:
            if cluster_rules.is_pair_mergeable(node1, node2):
                node2cid[node1] = cluster_counter
                node2cid[node2] = cluster_counter
                cid2cluster[cluster_counter] = cluster_rules.build_cluster(node1, node2)
                cluster_counter += 1
        elif c2 is None:
            added = cid2cluster[c1].add_element_if_allowed(node2)
            if added:
                node2cid[node2] = c1
        elif c1 is None:
            added = cid2cluster[c2].add_element_if_allowed(node1)
            if added:
                node2cid[node1] = c2
        # If 2 nodes already belong to the same cluster, there is nothing to do.
        elif c1 != c2:
            cmin = min(c1, c2)
            cmax = max(c1, c2)
            added = cid2cluster[cmin].merge_cluster(cid2cluster[cmax])
            if added:
                for node in cid2cluster[cmax].nodes():
                    node2cid[node] = cmin
                del cid2cluster[cmax]
    if compare_previous_limit:
        old_clusters_copy = {eid: frozenset(element for elements in source2element.values() for element in elements) for eid, source2element in eid2source2element.items()}
        eid2source2element.clear()
        new_cluster_copy = {cid: frozenset(cluster.nodes()) for cid, cluster in cid2cluster.items()}
        res = compare_clusters(old_clusters_copy, new_cluster_copy, compare_previous_limit)
        del old_clusters_copy
        del new_cluster_copy
    else:
        eid2source2element.clear()
        res = 0
    for eid, cluster in cid2cluster.items():
        for node in cluster.nodes():
            eid2source2element[eid][node.source].add(node)
    return res


class WeightedEdge:
    """
    Weighted edge, 1st and 2nd
    """
    def __init__(self, node_a, node_b, weight:float):
        self.weight = weight
        self.node1 = min(node_a, node_b)
        self.node2 = max(node_a, node_b)

    def __repr__(self):
        return "%s --- %s [%f]" % (self.node1, self.node2, self.weight)

    def __eq__(self, other):
        return self.node1 == other.node1 and self.node2 == other.node2

    def __hash__(self):
        return hash(self.node1) + hash(self.node2)



def nodes_flattener(source2elements):
    """
    Flatten nodes grouped in source2elements
    >>> nodes_flattener({10: {7,3,8}, 20: {6,4,2}})
    [2, 3, 4, 6, 7, 8]
    """

    return sorted(element for elements in source2elements.values() for element in elements)

def build_all_pairs_generic(clusters, nodes_retriever=nodes_flattener):
    """
    Build all pairs of nodes in same cluster.
    :param: clusters: list of clusters
    :param: nodes_retriever: method that retrieve nodes from a cluster
    >>> list(build_all_pairs_generic([[4,1,3,2], set([7,9,8])], lambda cluster:cluster))
    [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (7, 8), (7, 9), (8, 9)]
    >>> list(build_all_pairs_generic([{30:{100,200}},{10: {7,3,8}, 20: {6,4,2}}]))
    [(100, 200), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (3, 4), (3, 6), (3, 7), (3, 8), (4, 6), (4, 7), (4, 8), (6, 7), (6, 8), (7, 8)]

    """
    for cluster in clusters:
        all_nodes_in_cluster = sorted(nodes_retriever(cluster))
        for pair in itertools.combinations(all_nodes_in_cluster, 2):
            yield pair


def add_edges_bipartite(edges, source2sas1, source2sas2, weight:int):
    """
    Add edges between all pairs from sas1 and sas2
    :param edges: 
    :param sas1: 
    :param sas2: 
    :param weight: 
    :return: 
    """
    for source1, source2 in itertools.product(source2sas1.keys(), source2sas2.keys()):
        if source1 != source2:
            for sa1, sa2 in itertools.product(source2sas1[source1], source2sas2[source2]):
                edges.append(WeightedEdge(sa1, sa2, weight))


def add_group_edges(edges, sas_groups, weight: int):
    """
    Add edges between all elements in sas_groups (ie full graph)
    :param edges:
    :param sas_groups:
    :param weight:
    :param exclude_rule:
    :param group_all if True, group everything in a single clique, otherwise make separate cliques per group
    :return:
    """
    for sas in sas_groups:
        for sa_pair in itertools.combinations(sas, 2):
            edges.append(WeightedEdge(sa_pair[0], sa_pair[1], weight))


ClusterDifference = collections.namedtuple('ClusterDifference',
                                           'original_pairs added_pairs deleted_pairs sample_added sample_deleted')

def _build_flat_clusters(eid2source2elements:dict):
    """
    Build a list of clusters, represented as sorted tuple of elements
    >>> eid2source2elements = {1: {'s1':  {10,20},'s2':{30}}, 2: {'s1': {10}}, 3: {'s4': {50,60,70,80}}}
    >>> _build_flat_clusters(eid2source2elements)
    {(50, 60, 70, 80), (10,), (10, 20, 30)}
    """
    res = set()
    for source2elements in eid2source2elements.values():
        res.add(tuple(sorted(element for elements in source2elements.values() for element in elements)))
    return res


def cluster_differences(eid2source2element_old: dict, eid2source2element_new: dict,
                        sample_function=lambda x: x, overlapping_cluster=False) -> ClusterDifference:
    """
    Compute difference between 2 clustering. In particular, detect new and deleleted pairs
    :param overlapping_cluster: True if cluster are overlapping. Notice that alignment cluster are not overlapping per-se:
    there may be different virtual atts from an original att that pertain to different cluster, but technically they are
    different.
    
    >>> eid2source2element1 = {1: {'s1': {10,20,30}, 's2':  {40,50}}, \
                               2: {'s1': {60, 70}}, \
                               3: {'s3': {80}}, \
                               4: {'s5': {100,110}}}
    >>> eid2source2element2 = {9: {'s1': {10,20}, 's2': {40,50}}, \
                               5: {'s1': {60, 30, 70}}, \
                               6: {'s3': {80, 90}}, \
                               4: {'s5': {100,110}}}
    >>> cluster_differences(eid2source2element1, eid2source2element2)
    ClusterDifference(new_pairs=3, deleted_pairs=4, new_pairs_percent=0.25, deleted_pairs_percent=0.3333333333333333)
    """
    clusters_old = _build_flat_clusters(eid2source2element_old)
    clusters_new = _build_flat_clusters(eid2source2element_new)
    total_pairs_old = sum(bdsa_utils.nb_pairs_in_cluster(len(cluster)) for cluster in clusters_old)

    # Now we need to compute pairs of elements in same cluster, and check if some pairs are new and/or some is old.
    # If clusters are non-overlapping, we can throw away identical clusters, as they share exactly the same pairs.
    clusters_only_old = clusters_old if overlapping_cluster else clusters_old - clusters_new
    clusters_only_new = clusters_new if overlapping_cluster else clusters_new - clusters_old
    old_pairs = set().union(*(set(itertools.combinations(cluster, 2)) for cluster in clusters_only_old))
    new_pairs = set().union(*(set(itertools.combinations(cluster, 2)) for cluster in clusters_only_new))
    added_pairs = new_pairs - old_pairs
    deleted_pairs = old_pairs - new_pairs
    sample_added = stats_utils.safe_sample(added_pairs, 300)
    sample_deleted = stats_utils.safe_sample(deleted_pairs, 300)

    return ClusterDifference \
        (total_pairs_old, len(added_pairs), len(deleted_pairs), sample_added, sample_deleted)


def compare_clusters(cl1:dict, cl2:dict, min_similarity_to_continue=0.95):
    """
    Compare 2 clustering of same data. Typically used to decide when to stop iteration
    If ratio of element in identical clusters are less than min_similarity_to_continue, then stop and return this ratio.
        - For instance, [12, 345, 67, 8] and [1, 2345, 67, 8] have a ratio of 3/8 (678 over 1,2,3,4,5,6,7,8)
    This method is used to check whether 2 clusters are enough similar, so if with this quick analysis they are quite different,
    then we should stop in order to avoid losing too much time.
    If ratio is > min_similarity_to_continue, then continues:
    - build pair of "correspondent clusters", from cl1 and cl2, i.e.:
        - compute jaccard containment similarity between all cluster pairs that share at least an element
        - take the pair with biggest JS --> label as 'correspondent' clusters
        - continue iteratively last point (unless a cluster is already in a pair)
    - compute score as #elements in cl1 that are in their correspondent cluster in cl2 / total elements

    :param cl1: dict of FROZEN set
    :param cl2: same
    :param min_similarity_to_continue: see description
    :return:
    >>> cl1 = {100:frozenset([1,2,3]), 200:frozenset([4,5,6]), 300:frozenset([7,8,9]), 400:frozenset([15,16])}
    >>> cl2 = {500:frozenset([1,2]), 600:frozenset([3,4,5,6]), 700:frozenset([7,8,9,10]), 800:frozenset([15,16])}
    >>> res = compare_clusters(cl1, cl2, 0.05)
    >>> round(res, 3)
    0.833
    >>> cl1
    {100: frozenset({1, 2, 3}), 200: frozenset({4, 5, 6}), 300: frozenset({8, 9, 7}), 400: frozenset({16, 15})}
    >>> cl2
    {500: frozenset({1, 2}), 600: frozenset({3, 4, 5, 6}), 700: frozenset({8, 9, 10, 7}), 800: frozenset({16, 15})}
    """
    len1 = sum(len(group) for group in cl1.values())
    len2 = sum(len(group) for group in cl2.values())
    # Join clusters
    intersect = set(cl1.values()) & set(cl2.values())
    len_intersec = sum(len(group) for group in intersect)
    similarity_rough = len_intersec / max(len1, len2) if len1 != 0 and len2 != 0 else 1
    if similarity_rough < min_similarity_to_continue:
        return similarity_rough

    # Now compare the remaining
    cl1_filtered = {key: values for key, values in cl1.items() if values not in intersect}
    cl2_filtered = {key: values for key, values in cl2.items() if values not in intersect}
    cl2_inverted = collections.defaultdict(set)
    for cid, elements in cl2_filtered.items():
        for element in elements:
            cl2_inverted[element].add(cid)
    element_pairs = []
    for cid1, elements in cl1_filtered.items():
        counter_cid2 = collections.Counter()
        for element in elements:
            counter_cid2.update(cl2_inverted[element])
        for cid2, occs in counter_cid2.items():
            # Jaccard similarity
            element_pairs.append((cid1, cid2, occs / max(len(elements), len(cl2_filtered[cid2])), occs))
    counted_c1 = set()
    counted_c2 = set()
    total_correct = len_intersec
    for pair in sorted(element_pairs, key=lambda pair: pair[2], reverse=True):
        if pair[0] not in counted_c1 and pair[1] not in counted_c2:
            total_correct += pair[3] # TODO fix
            counted_c1.add(pair[0])
            counted_c2.add(pair[1])
    return total_correct / max(len1, len2) if len1 != 0 and len2 != 0 else 1