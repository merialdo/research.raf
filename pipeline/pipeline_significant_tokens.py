import collections
import itertools
import math

from tqdm import tqdm

from model.bdsa_data import BdsaData
from model.bdsa_data_transformed import BdsaDataTransformed
from pipeline.pipeline_abstract import AbstractPipeline
from pipeline.pipeline_common import ClusteringOutput
from utils import bdsa_utils

from config.bdsa_config import _config_

def identify_significant_tokens(clustering_output:ClusteringOutput) -> dict:
    """
    From each cluster, identify the significant tokens. I.e, tokens whose presence in an attribute value implies
    presence in other attribute values for linked pages
    :param sa_clusters:
    :return:
    """
    clustering_output.bdsa_data.remove_all_generated()
    source2ids = clustering_output.source2pids
    cid2tokens = collections.defaultdict(set)
    for cid, source2sas in tqdm(clustering_output.sa_clusters.items(), desc='Identify significant tokens...'):
        #Identify all entities (page IDs) potentially involved
        pid_count = collections.Counter()
        for source in source2sas.keys():
            pid_count.update(source2ids[source])
        pids = set(pid for pid, occs in pid_count.items() if occs >= 2)
        sa2tokens = _compute_tokens_for_sa(source2sas, clustering_output.bdsa_data.get_transformed_data())

        sa_pair2occs, token2sa_pair2event = _analyze_tokens_in_cluster(clustering_output, pids, source2sas, sa2tokens)
        # Now compute M.I.
        sum_occurrences = sum(sa_pair2occs.values())
        for token, sa_pair2presence in token2sa_pair2event.items():
            mi_token, nb_cooccurrences = _compute_mutual_information(sa_pair2occs, sa_pair2presence, sum_occurrences)
            if mi_token >= _config_.get_min_mi() \
                    and sum(1 for occs in nb_cooccurrences if occs >= _config_.get_min_page_pair()) >= _config_.get_min_sa_pair():
                cid2tokens[cid].add(token)
    cid2tokens_final = _filter_cid2token_list(cid2tokens, clustering_output.sa_clusters)

    return cid2tokens_final


def _filter_cid2token_list(cid2tokens, sa_clusters:dict, min_significant_tokens=_config_.get_min_significant_tokens()):
    """
    Filter clusters used for generating attributes (ie those clusters would not be used anymore to generate attributes).
    Remove cluster with less than K tokens, and if 2 clusters are related to the same tokens, then remove one of them.
    :param cid2tokens:
    :return:
    >>> cid2tokens = {0: {1}, 1: {3,4,5}, 2: {4,5}, 3:{3,4,5}, 7: {1}, 8:{3,4,5}, 9:{3,4,10}}
    >>> sa_clusters = {1: {10,11,12}, 3: {11,12,13,14,15,16}, 8:{10,13,14}}
    >>> _filter_cid2token_list(cid2tokens, sa_clusters, 3)
    {3: frozenset({3, 4, 5}), 9: frozenset({10, 3, 4})}
    """
    tokens2cid = {}
    cid2tokens_final = {}
    for current_cid, tokens in cid2tokens.items():
        if len(tokens) >= min_significant_tokens:
            token_frozen = frozenset(tokens)
            conflicting_cid = tokens2cid.get(token_frozen, None)
            if token_frozen in tokens2cid:
                if len(sa_clusters[current_cid]) > len(sa_clusters[conflicting_cid]):
                    del cid2tokens_final[conflicting_cid]
                    cid2tokens_final[current_cid] = token_frozen
                    tokens2cid[token_frozen] = current_cid
            else:
                cid2tokens_final[current_cid] = token_frozen
                tokens2cid[token_frozen] = current_cid
    return cid2tokens_final


def _compute_mutual_information(sa_pair2occs, sa_pair2presence, sum_occurrences):
    """
    Compute mutual information for a token between a pair of attributes
    :param sa_pair2occs:
    :param sa_pair2presence:
    :param sum_occurrences:
    :return: the mutual information AND a support value, nb_cooccurrences: [3,4,5] means the token co-coccurred in 3 pages
    for an attribute pair, 4 pages for another and
    """
    mi_token_sum = 0
    nb_cooccurrences = []
    for sa_pair, presences in sa_pair2presence.items():
        log_sa_pair_occs = math.log(sa_pair2occs[sa_pair])
        # This sum should by eventually divided by sa_pair2occs[sa_pair] and multiplied by the same
        # for weighted average, so it clears out
        nb_instances_with_token_s1 = sum(occs for event, occs in presences.items() if event[0])
        nb_instances_with_token_s2 = sum(occs for event, occs in presences.items() if event[1])
        nb_instances_without_token_s1 = sa_pair2occs[sa_pair] - nb_instances_with_token_s1
        nb_instances_without_token_s2 = sa_pair2occs[sa_pair] - nb_instances_with_token_s2
        for event, occs in presences.items():
            mi_token_sum += occs * (log_sa_pair_occs + math.log(occs /
                                                                ((nb_instances_with_token_s1 if event[
                                                                    0] else nb_instances_without_token_s1) *
                                                                 nb_instances_with_token_s2 if event[
                                                                    1] else nb_instances_without_token_s2)))
            if event == (True,True):
                nb_cooccurrences.append(occs)
    mi_token = mi_token_sum / sum_occurrences
    return mi_token, nb_cooccurrences


def _analyze_tokens_in_cluster(clustering_output, pids, source2sas, sa2tokens):
    """
    For each token and attribute pair, here we store the 4 possible events, each with nb occurrences of event
    - token in both (True,True)
    - token in first/second (True,False) or (False, True)
    - token in noone (False,False)
    :param clustering_output:
    :param pids: product IDs present in cluster
    :param source2sas: sas of clusters, grouped by source
    :param sa2tokens: all tokens of each sa in the cluster
    :param transformer:
    :return: sa_pair2occs, total occurrence of each attribure pair, AND token2sa_pair2event (the one discussed above)
    """
    token2sa_pair2event = collections.defaultdict(bdsa_utils.dd_counter_generator)
    sa_pair2occs = collections.Counter()
    transformer = clustering_output.bdsa_data.get_transformed_data()
    # For each entity, find occurrences of tokens in all pairs
    for pid in pids:
        # Involved sources are sources that provide data for this entity/pid AND that have at least 1 source attribute of this cluster
        sources_involved = sorted(clustering_output.page_clusters[pid].keys() & source2sas.keys())
        for source1, source2 in itertools.combinations(sources_involved, 2):
            for page1, page2 in itertools.product(clustering_output.page_clusters[pid][source1],
                                                  clustering_output.page_clusters[pid][source2]):
                specs_in_page1 = transformer.get_sa2value_for_page(page1)
                specs_in_page2 = transformer.get_sa2value_for_page(page2)
                # Only sa in this cluster
                specs_involved_page1 = {sa: value for sa, value in specs_in_page1.items() if sa in source2sas[source1]}
                specs_involved_page2 = {sa: value for sa, value in specs_in_page2.items() if sa in source2sas[source2]}
                for sa1, sa2 in itertools.product(specs_involved_page1, specs_involved_page2):
                    sa_pair2occs[(sa1, sa2)] += 1
                    val1 = specs_involved_page1[sa1]
                    val2 = specs_involved_page2[sa2]
                    # Analyze only tokens present in domain of at least 1 attribute
                    for token in sa2tokens[sa1] | sa2tokens[sa2]:
                        token2sa_pair2event[token][(sa1, sa2)][(token in val1, token in val2)] += 1
    return sa_pair2occs, token2sa_pair2event


def _compute_tokens_for_sa(source2sas, transformer):
    """
    :param source2sas:
    :param transformer:
    :return: for each sa, token it contains in its value domain
    """
    sa2tokens = collections.defaultdict(set)
    # Identify domain token for each sa
    for source, sas in source2sas.items():
        for sa in sas:
            for value_tokenized in transformer.get_transformed_value2occs(sa).keys():
                sa2tokens[sa].update(value_tokenized)
    return sa2tokens


def generate_attributes_from_tokens(bdsa_data:BdsaData, cluster2significant_tokens:dict, cid2name:dict,
                                    transformer:BdsaDataTransformed):
    """
    For each attribute containing some significant tokens, generate a correspondent attribute with only those tokens.
    :param bdsa_data:
    :param cluster2significant_tokens:
    :return:
    """
    for source, pages in tqdm(bdsa_data.source2pages.items(), desc='Build generated attributes for significant tokens...'):
        for page in pages:
            for sa, value_tokenized in transformer.get_sa2value_for_page(page).items():
                for cid, sign_tokens in cluster2significant_tokens.items():
                    intersection = sign_tokens & value_tokenized
                    if len(intersection) > 0:
                        yield bdsa_data.GeneratedAttributeOccurrence(sa.name,
                                                        ' '.join(intersection), page, 'mi_%s_%d' % (cid2name[cid], cid))

class PipelineSignificantTokens(AbstractPipeline):
    """
    This pipeline extract significant tokens from already aligned attributes, then creates generated  attributes
    with only significant tokens. S.T. are token useful for computing matches
    """
    def __init__(self):
        self.debug_stats = None
        # TODO implement a stop condition for iterations.
        self.continue_iterations = True

    def run(self, data):
        result = {}
        self.debug_stats = data[1]
        for cat, clustering_output in data[0].items():
            result[cat] = self.run_category(cat, clustering_output)

        return result, self.debug_stats

    def run_category(self, cat:str, clustering_output:ClusteringOutput) -> ClusteringOutput:
        """
        Rune the pipeline for a specific category
        :param cat:
        :param clustering_output:
        :return:
        """
        cluster2significant_tokens = identify_significant_tokens(clustering_output)
        cid2name = clustering_output.find_name_for_clusters()
        bdsa_data = clustering_output.bdsa_data
        # Transformer instance variable is eliminated from bdsa_data when some generated data is built, however we keep
        # it as a local variable and delete it at the end of matching.
        transformer = bdsa_data.get_transformed_data()
        attributes_generator = generate_attributes_from_tokens(bdsa_data, cluster2significant_tokens,
                                                               cid2name, transformer)
        custom_filter = lambda sa, bdsa_data: len(bdsa_data.sa2value2occs[sa]) >= _config_.get_min_distinct_values_generated_mi()
        bdsa_data.launch_attribute_generation(attributes_generator, custom_filter)
        return clustering_output

    def name(self):
        return "PipelineSignificantTokens"

    def need_input(self):
        return True

    def need_output(self):
        return True


