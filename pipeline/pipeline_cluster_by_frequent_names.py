import collections
import copy

from tqdm import tqdm

import pipeline.cluster_utils
from model.clusters import SaClusterRules, InterSourceClusterRules, FreeClusterRules
from pipeline import pipeline_common
from pipeline.pipeline_abstract import AbstractPipeline
from pipeline.pipeline_common import ClusteringOutput
from model.bdsa_data import BdsaData
from utils import string_utils

MIN_OCCURRENCES = 7
MAX_ATTRIBUTES = 7

class PipelineClusterByFrequentNames(AbstractPipeline):
    """
    Pipeline step that groups clusters with same most frequent attribute name
    """

    def run(self, data):
        result_output = {}
        self.debug_stats = data[1]
        for cat, clustering_output in data[0].items():
            result_output[cat] = self.compute_group_alignment(clustering_output, cat)

        return result_output, self.debug_stats

    def name(self):
        return "NameClusteringBaseline"

    def need_input(self):
        return True

    def need_output(self):
        return True

    def __init__(self):
        self.debug_stats = collections.defaultdict(dict)

    def compute_group_alignment(self, output: ClusteringOutput, cat: str) -> ClusteringOutput:

        name2cluster = collections.defaultdict(set)
        name2isolated = collections.defaultdict(set)
        edges = []
        for cid, source2sa in tqdm(output.sa_clusters.items(), desc='Find common name in clusters...'):
            name_count = collections.Counter()
            flatten_sas = [sa for sas in source2sa.values() for sa in sas]
            pipeline.cluster_utils.add_group_edges(edges, [flatten_sas], 2)
            name_count.update([string_utils.folding_using_regex(sa.get_original_name()) for sa in flatten_sas])
            nb_sas = len(flatten_sas)
            name_ratio = {saname: occs / nb_sas for saname, occs in name_count.items()}
            for name, occs in name_count.items():
                if occs >= 2 and name_ratio[name] >= 0.33:
                    name2cluster[name].add(cid)

        for isa in tqdm(output.sa_isolated, desc='Build map name 2 isolated...'):
            if not isa.is_generated():
                name2isolated[string_utils.folding_using_regex(isa.get_original_name())].add(isa)

        for isolated in name2isolated.values():
            pipeline.cluster_utils.add_group_edges(edges, [isolated], 1)

        for name, clusters in tqdm(name2cluster.items(), desc='Grouping clusters...'):
            sas_from_clusters = []
            for cid in clusters:
                any_sa_list = next(iter(output.sa_clusters[cid].values()))
                any_sa = next(iter(any_sa_list))
                sas_from_clusters.append(any_sa)
            isolated_group = [any_sa]  # We put together a random SA from the cluster with a random isolated elements
            if name in name2isolated:
                isolated_group.append(next(iter(name2isolated[name])))
            pipeline.cluster_utils.add_group_edges(edges, [sas_from_clusters, isolated_group], 1)
        pipeline.cluster_utils.partition_using_agglomerative(edges, output.sa_clusters,
                                                             FreeClusterRules())
        pipeline_common.remove_non_master_attributes(output.sa_clusters)
        return output
