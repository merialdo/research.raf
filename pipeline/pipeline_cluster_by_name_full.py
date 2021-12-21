import collections

import itertools
from tqdm import tqdm

import pipeline.cluster_utils
from model.bdsa_data_transformed import BdsaDataTransformed
from model.clusters import FreeClusterRules, SaClusterRules
from model.datamodel import SourceAttribute
from pipeline import cluster_utils
from pipeline.pipeline_abstract import AbstractPipeline
from pipeline.pipeline_common import ClusteringOutput
from utils import bdsa_utils, string_utils, tokenize_utils, prob_utils
from test.test_utils import tsa, ts

MIN_JACCARD_CONTAINMENT = 0.8


class PipelineClusterByNameFull(AbstractPipeline):
    """
    Put together atts with same name, merging clusters obtained with other methods.
    Similar to cluster_by_frequent_names by without any thresholds on frequence of name in group

    """

    def need_input(self):
        return True

    def need_output(self):
        return True

    def name(self):
        return "NameClusteringFull"

    def __init__(self, _same_page_rule=False, exclude_splitted_original=False, blacklist_names=False):
        self.same_page_rule = _same_page_rule
        self.exclude_splitted_original = exclude_splitted_original
        self.blacklist_names = blacklist_names

    """
    Pipeline step that groups clusters with same most frequent attribute name
    """

    def run(self, data):
        result_output = {}
        self.debug_stats = data[1]
        for cat, clustering_output in data[0].items():
            result_output[cat] = compute_group_alignment(clustering_output, cat,
                                                         self.same_page_rule, self.exclude_splitted_original,
                                                         self.blacklist_names)

        return result_output, self.debug_stats

def compute_group_alignment(output: ClusteringOutput, cat: str, same_page_rule:bool,
                            exclude_original_with_clustered_generated:bool,blacklist_names:bool) -> ClusteringOutput:
    """

    :param output:
    :param cat:
    :param exclude_original_with_clustered_generated: if an original attribute has some virtual attribute that has been 
    clustered, then exclude it from merge
    :param blacklist_names: exclude any attribute name for which it exists at least a source attribute non-homogeneous
    :return:
    S1              A   B   C
                    |     /
    S2              D   B   C
                    |   |
    S3              F   G
    >>> s1 = ts('1'); s2 = ts('2'); s3 = ts('3')
    >>> cout = ClusteringOutput()
    >>> cout.sa_clusters.update({1:{s1 : {tsa('a', '1')}, s2: {tsa('d', '2')}, \
s3: {tsa('f','3')}}, 2:{s1 : {tsa('c', '1')}, s2: {tsa('b', '2')}, s3: {tsa('g','3')}}})
    >>> cout.sa_isolated = {tsa('b', '1'): {}, tsa('c', '2'): {}}
    >>> cout.bdsa_data.sa2size = {tsa('a', '1'):1,tsa('b', '1'):1,tsa('c', '1'):1,tsa('d', '2'):1,\
tsa('b', '2'):1,tsa('c', '2'):1,tsa('f', '3'):1,tsa('g', '3'):1}
    >>> otp = compute_group_alignment(cout, None, False, False, True)
    >>> bdsa_utils.dict_printer(otp.sa_clusters)
    """

    name2cid = collections.defaultdict(set)
    name2sas_isolated = collections.defaultdict(set)
    edges = []

    virtual_attributes_aligned = collections.defaultdict(set)

    for cid, source2sa in tqdm(output.sa_clusters.items(), desc='Find common name in clusters...'):
        flatten_sas = [sa for sas in source2sa.values() for sa in sas]
        pipeline.cluster_utils.add_group_edges(edges, [flatten_sas], 2)
        for sa in flatten_sas:
            name_token = tokenize_utils.value2token_set(sa.get_original_name())
            if not sa.is_generated() or not exclude_original_with_clustered_generated:
                # Align only original attributes?
                name2cid[name_token].add(cid)
             # If it is generated than apply rules
            if sa.is_generated():
                virtual_attributes_aligned[sa.get_original_attribute()].add(sa)

    names_to_blacklist = set()
    for sa in output.sa_isolated:
        # Non-aligned generated attributes are ignored
        if not sa in virtual_attributes_aligned.keys():
            name_token = tokenize_utils.value2token_set(sa.name)
            name2sas_isolated[name_token].add(sa)
        if sa in virtual_attributes_aligned.keys() and blacklist_names:
            names_to_blacklist.add(sa.name)


    transformed = output.bdsa_data.get_transformed_data()

    for name in name2sas_isolated.keys() | name2cid.keys():
        for cid1, cid2 in itertools.combinations(name2cid[name], 2):
            ctok1 = build_cluster_token_domain(cid1, output, transformed).keys()
            ctok2 = build_cluster_token_domain(cid2, output, transformed).keys()
            jc = prob_utils.compute_jaccard_similarity(ctok1, ctok2, True)
            if jc > MIN_JACCARD_CONTAINMENT:
                cluster_utils.add_edges_bipartite(edges, output.sa_clusters[cid1], output.sa_clusters[cid2], jc)

        for sa1, sa2 in itertools.combinations(name2sas_isolated[name], 2):
            jc = prob_utils.compute_jaccard_similarity(build_attribute_token_domain(sa1, transformed).keys(),
                                                       build_attribute_token_domain(sa2, transformed).keys(), True)
            if jc > MIN_JACCARD_CONTAINMENT:
                edges.append(cluster_utils.WeightedEdge(sa1, sa2, jc))

        for sa, cid in itertools.product(name2sas_isolated[name], name2cid[name]):
            cdom = build_cluster_token_domain(cid, output, transformed).keys()
            jc = prob_utils.compute_jaccard_similarity(cdom, build_attribute_token_domain(sa, transformed).keys(), True)
            if jc > MIN_JACCARD_CONTAINMENT:
                for sa_cluster in cluster_utils.nodes_flattener(output.sa_clusters[cid]):
                    edges.append(cluster_utils.WeightedEdge(sa, sa_cluster, jc))

    merge_rule = SaClusterRules(output.bdsa_data.sa2urls) if same_page_rule else FreeClusterRules()
    pipeline.cluster_utils.partition_using_agglomerative(edges, output.sa_clusters, merge_rule)
    #pipeline_common.remove_non_master_attributes(output.sa_clusters)
    return output

def build_cluster_token_domain(cid, output:ClusteringOutput, transformed:BdsaDataTransformed):
    res = collections.Counter()
    for sas in output.sa_clusters[cid].values():
        for sa in sas:
            res.update(build_attribute_token_domain(sa, transformed).keys())
    return res

def build_attribute_token_domain(sa:SourceAttribute, transformed:BdsaDataTransformed):
    domain = collections.Counter()
    for value in transformed.get_transformed_value2occs(sa).keys():
        for token in value:
            domain[token] += 1
    return domain



def jaccard(c1:collections.Counter, c2:collections.Counter, containment:bool):
    """
    Returns if JC between 2 counters is bigger than MIN_JACCARD_CONTAINMENT
    :param c1: 
    :param c2: 
    :return: 
    """
    denominator = min(sum(c1.values()), sum(c2.values())) if containment else sum((c1 | c2).values())
    return sum((c1 & c2).values()) / denominator


