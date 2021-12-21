import collections
import copy

import pipeline.cluster_utils
from model.clusters import SaClusterRules
from pipeline import pipeline_common
from pipeline.pipeline_abstract import AbstractPipeline
from pipeline.pipeline_common import ClusteringOutput
from model.bdsa_data import BdsaData

MIN_OCCURRENCES = 7
MAX_ATTRIBUTES = 7

class PipelineComputeWeights(AbstractPipeline):

    def run(self, data):
        result = {}
        self.debug_stats = data[1]
        for cat, clustering_output in data[0].items():
            result[cat] = self.compute_linkage_and_alignment(clustering_output, cat)

        return result, self.debug_stats

    def name(self):
        return "NameClusteringBaseline"

    def need_input(self):
        return True

    def need_output(self):
        return True

    def __init__(self, use_rare_values=True):
        self.debug_stats = None
        self.use_rare_values = use_rare_values

    def compute_linkage_and_alignment(self, clustering_output:ClusteringOutput, cat:str) -> ClusteringOutput:

        name2sa = collections.defaultdict(set)
        frequentvalue2sa = collections.defaultdict(set)

        # Initialize page clusters as the one provided in input
        # Some other infos are passed directly to output, as they are useful in analysis
        bdsa_data = clustering_output.bdsa_data
        clustering_output.page_clusters = copy.deepcopy(bdsa_data.pid2source2pages)
        for sa in bdsa_data.sa2size.keys():
            name2sa[sa.name].add(sa)
            if self.use_rare_values:
                # TODO is it ok to use transformed data here?
                for value, occs in bdsa_data.get_transformed_data().get_transformed_value2occs(sa).items():
                    value_frequent_in_att = occs > MIN_OCCURRENCES or occs / bdsa_data.sa2size[sa] > 0.5
                    value_rare_generally = bdsa_data.get_transformed_data().nb_distinct_occurrences_attributes(value) <= MAX_ATTRIBUTES
                    if value_frequent_in_att and value_rare_generally:
                        frequentvalue2sa[value].add(sa)
        edges = []
        pipeline.cluster_utils.add_group_edges(edges, name2sa.values(), 2)
        if self.use_rare_values:
            pipeline.cluster_utils.add_group_edges(edges, frequentvalue2sa.values(), 1)
        pipeline.cluster_utils.partition_using_agglomerative(edges, clustering_output.sa_clusters,
                                                             SaClusterRules(bdsa_data.sa2urls))
        return clustering_output
