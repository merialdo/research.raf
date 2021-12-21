import collections
import copy

import pipeline.cluster_utils
from model.clusters import SaClusterRules
from pipeline import pipeline_common
from pipeline.pipeline_abstract import AbstractPipeline
from pipeline.pipeline_common import ClusteringOutput
from pipeline.cluster_utils import WeightedEdge
from model.bdsa_data import BdsaData
from test.test_utils import tsa
from utils import bdsa_utils
from utils.blocking_graph import MetablockingGraph

MAX_VALUE_FREQUENCY = 0.2

MIN_OCCURRENCES = 7
MAX_ATTRIBUTES = 7

class BaselineDomainAttributes(AbstractPipeline):
    """
    Join attributes with similar domains
    """

    def run(self, data):
        result = {}
        self.debug_stats = data[1]
        if self.aggregate_other_clustering:
            for cat, output in data[0].items():
                result[cat] = self._aggregate_current_cluster_with_domain_based(output, cat)
        else:
            for cat, output in data[0].items():
                result[cat] = self.full_alignment_domain_based(output, cat)

        return result, self.debug_stats

    def name(self):
        return "NameClusteringBaseline"

    def need_input(self):
        return True

    def need_output(self):
        return True

    def __init__(self, _aggregate_other_clustering=False, _threshold_frequency=MAX_VALUE_FREQUENCY):
        """

        :param _aggregate_other_clustering: aggregate results with previous clustering
        """
        self.aggregate_other_clustering = _aggregate_other_clustering
        self._threshold_frequency = _threshold_frequency
        self.debug_stats = None

    def full_alignment_domain_based(self, output:ClusteringOutput, cat:str) -> ClusteringOutput:
        bdsa_data = output.bdsa_data
        graph = self._build_graph_similar_domains(bdsa_data)

        clustering_output = ClusteringOutput()
        clustering_output.bdsa_data = bdsa_data
        pipeline.cluster_utils.partition_using_agglomerative(graph, clustering_output.sa_clusters,
                                                             SaClusterRules(bdsa_data.sa2urls))
        return clustering_output

    def _build_graph_similar_domains(self, bdsa_data):
        """
        Build list of edges of attribute pairs with similar domains
        :param bdsa_data:
        :return:
        """
        value2source2sas = _build_inverse_value_maps(bdsa_data.sa2size.keys(), bdsa_data.get_transformed_data()
                                                     .get_transformed_value2occs,
                                                     bdsa_data.get_transformed_data().nb_distinct_occurrences_attributes,
                                                     threshold=self._threshold_frequency)
        mb_graph = MetablockingGraph(2)
        mb_graph.add_full_clique(value2source2sas.values(), 1)
        graph = []
        candidates_by_source = mb_graph.get_candidates()
        for sa_pair in (sa_pair for sa_pairs in candidates_by_source.values() for sa_pair in sa_pairs):
            score = _compute_attribute_pair_score(bdsa_data.get_transformed_data().get_transformed_value2occs,
                                                  bdsa_data.sa2size, sa_pair[0], sa_pair[1])
            graph.append(WeightedEdge(sa_pair[0], sa_pair[1], score))
        return graph

    def _aggregate_current_cluster_with_domain_based(self, output:ClusteringOutput, cat:str):
        """
        Aggregate cluster already present with new clusters
        :param output:
        :param cat:
        :return:
        """
        data = output.bdsa_data
        graph = self._build_graph_similar_domains(data)
        pipeline.cluster_utils.aggregate_clusters(graph, output.sa_clusters, 0.9, SaClusterRules(data.sa2urls), None)
        return output



def _compute_attribute_pair_score(value2occs_function, sa2size, sa1, sa2):
    """

    :param value2occs_function:
    :param sa2size:
    :param pair:
    :return:
    >>> sa2size = {tsa('a'): 100, tsa('b'): 10, tsa('c'): 5}
    >>> sa2val2occs = {tsa('a'): {10: 10, 20:1, 40:8}, tsa('b'): {10:5, 30:2, 40:10}, tsa('c'): {10:1, 40:1, 30:2}}
    >>> _compute_attribute_pair_score(lambda sa: sa2val2occs[sa], sa2size, tsa('a'), tsa('b'))
    0.0018
    """
    values1 = value2occs_function(sa1)
    values2 = value2occs_function(sa2)
    intersection = sum(min(value2occs_function(sa1)[value] / sa2size[sa1], value2occs_function(sa2)[value] / sa2size[sa2]) for value in values1.keys() & values2.keys())
    return intersection / max(sa2size[sa1], sa2size[sa2])

def _build_inverse_value_maps(sas, values2occs_function, val2nb_source_attributes, threshold):
    """
    Build a map of value --> sas (grouped by source), only for values not too common.
    Useful to detect candidate attributes to compare
    :param bdsa_data:
    :return:
    >>> sas = [tsa('a'), tsa('b'), tsa('c')]
    >>> sa2val2occs = {tsa('a'): {10: 1, 20:1}, tsa('b'): {10:1, 30:2}, tsa('c'): {10:1, 40:1, 30:2}}
    >>> distinct_occs = {10:3, 20:1, 30:2, 40:1}
    >>> res = _build_inverse_value_maps(sas, lambda sa: sa2val2occs[sa], lambda  val: distinct_occs[val], threshold=0.7)
    >>> dict(res)
    {20: defaultdict(<class 'set'>, {SourceSpecifications(site='test_site', category='dummy', pages=None): \
SourceAttribute(source=SourceSpecifications(site='test_site', category='dummy', pages=None), name='a')}}), \
30: defaultdict(<class 'set'>, {SourceSpecifications(site='test_site', category='dummy', pages=None): \
{SourceAttribute(source=SourceSpecifications(site='test_site', category='dummy', pages=None), name='b'), \
SourceAttribute(source=SourceSpecifications(site='test_site', category='dummy', pages=None), name='c')}}), \
40: defaultdict(<class 'set'>, {SourceSpecifications(site='test_site', category='dummy', pages=None): \
{SourceAttribute(source=SourceSpecifications(site='test_site', category='dummy', pages=None), name='c')}})}

    """
    value2source2sas = collections.defaultdict(bdsa_utils.dd_set_generator)
    nb_source_atts = len(sas)
    for sa in sas:
        for value in values2occs_function(sa).keys():
            if val2nb_source_attributes(value) / nb_source_atts < threshold:
                value2source2sas[value][sa.source].add(sa)
    return value2source2sas
