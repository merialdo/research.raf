from difflib import SequenceMatcher

import collections

from config import constants
from config.bdsa_config import _config_
from math import *

from utils import string_utils, tokenize_utils, stats_utils

MIN_JACCARD_SIMILARITY = 0.9

def compute_attribute_equivalent_probability(value_matches: list, name1: str, name2: str, domain_full1: dict, domain_full2: dict,
                                             size1: int, size2: int, value2nb_attributes: dict, total_nb_attributes: int) -> float:
    """
    Compute the probability the 2 attributes are equivalent
    
    :return: 
    :return: 
    :param idf_match_weight: 
    :param value_matches:  group of 3-tuples with 2 values AND linkage accuracy
    :param node0: should contain value frequencies and number of values 
    :param node1: same
    :param idf: inverse document frequency of values
    :return: 
    """

    log_pos_posterior = 0
    log_neg_posterior = 0

    product_sizes = float(size1 * size2)
    sum_sizes = float(size1 + size2)

    # Domain of linked values. Values may have a weight < 1 if there are multiple values associated to a single product
    domain_linked1 = collections.defaultdict(float)
    domain_linked2 = collections.defaultdict(float)

    for v1, v2, weight_internal in value_matches:

        if _config_.compute_posterior_probability():
            neg_element, pos_element = _compute_probability_single_observation(domain_full1, domain_full2,
                                                                               product_sizes, sum_sizes, v1, v2, False)
            log_pos_posterior += (log(pos_element) * weight_internal if pos_element > 0 else float('-inf'))
            log_neg_posterior += (log(neg_element) * weight_internal if neg_element > 0 else float('-inf'))
        domain_linked1[v1] += weight_internal
        domain_linked2[v2] += weight_internal

    # Now define the a-priori. If one provides the IDF, we compute a-priori measuring similarity between values,
    # otherwise we use a fixed one.
    if value2nb_attributes is not None:
        common_values = {value: min(domain_full1[value] / float(size1), domain_full2[value] / float(size2)) for value in domain_full1.keys() & domain_full2.keys()}
        similarity_whole = sum(common_occs * 2 / float(value2nb_attributes(value)) for value, common_occs in common_values.items())
        nb_linkages = sum(v[2] for v in value_matches)

        common_linked_values = {value: min(domain_linked1[value], domain_linked2[value]) / float(nb_linkages) for value in domain_linked2.keys() & domain_linked1.keys()}
        similarity_linkage = sum(occs * 2 / float(value2nb_attributes(val)) for val, occs in common_linked_values.items())

        # Linkage similarity is weighted according to the number of values for products in linkage
        log_size_linkage = log(len(value_matches))
        # Global similairity is weighted according to the nb of occurences of biggest value
        log_size_whole = log(max(sum(domain_full1.values()), sum(domain_full2.values())))
        # Simplified weighted sum. Add default case if denominator is 0.
        similarity = similarity_linkage if log_size_whole == 0 else \
            (log_size_linkage / log_size_whole)*(similarity_linkage - similarity_whole) + similarity_whole
    else:
        similarity = _config_.get_default_apriori_equivalence_ratio()

    similarity_names = compute_jaccard_similarity(tokenize_utils.value2token_set(name1), tokenize_utils.value2token_set(
        name2))  #SequenceMatcher(None, name1, name2).ratio()
    prob_apriori_equivalence = _config_.get_similarity_names_weight() * similarity_names + (1 - _config_.get_similarity_names_weight()) * similarity
    #/ 2 if _config_.do_simplified_prior() else \
     #   similarity + _config_.get_similarity_names_weight() * (similarity_names - 0.5)

    prob_apriori_equivalence = min(1 - 1e-100, max(1e-100, prob_apriori_equivalence))
    prob = compute_bayes_probability_from_log(log_neg_posterior, log_pos_posterior, prob_apriori_equivalence)
    return prob, prob_apriori_equivalence


def _compute_probability_single_observation(domain1, domain2, product_2_sizes, sum_2_sizes, v1, v2, for_page_linkage):
    """
    Compute the probability of 2 values, given the domain of respective attributes
    :param domain1: 
    :param domain2: 
    :param product_2_sizes: 
    :param sum_2_sizes: 
    :param v1: 
    :param v2: 
    :param: for_page_linkage: true if doing page linkage, false if aligning attributes
    :return: 
    """

    # If the 2 atts are the same, the domain is the joint domain
    v0_freq_in_union_att_values = (domain1.get(v1, 0) + domain2.get(v1, 0)) / sum_2_sizes
    v1_freq_in_union_att_values = (domain1.get(v2, 0) + domain2.get(v2, 0)) / sum_2_sizes

    # Each value is independent in casual match, so we compute product of frequencies
    # Note that IF we are doing page linkage, we ALREADY suppose the 2 attributes are aligned, so we compute product
    # of JOINT domain. Otherwise for attribute linkage, we suppose attributes are unrelated
    neg_element = v0_freq_in_union_att_values * v1_freq_in_union_att_values \
        if for_page_linkage else domain1[v1] * domain2[v2] / product_2_sizes

    if v1 == v2 or compute_jaccard_similarity(v1, v2) >= MIN_JACCARD_SIMILARITY:
        pos_element_given_linkage = v0_freq_in_union_att_values * (1 -
                                                                   _config_.get_error_rate_per_value() * (
                                                                       2 - _config_.get_error_rate_per_value())
                                                                   * (1 - v0_freq_in_union_att_values))
    else:
        pos_element_given_linkage = _config_.get_error_rate_per_value() * (
            2 - _config_.get_error_rate_per_value()) \
                                    * v0_freq_in_union_att_values * v1_freq_in_union_att_values

    # we currently ignore linkage score. Suppose data we retrieved is correct.
    # pos_element = pos_element_given_linkage * linkage_score + \
    #               v0_freq_in_union_att_values * v1_freq_in_union_att_values * (1 - linkage_score)
    pos_element = pos_element_given_linkage
    return neg_element, pos_element

def compute_page_linkage_accuracy(all_datas):
    """
    Compute the accuracy of linkage between 2 pages 
    :param all_datas: 
    :param idf: 
    :return: 
    """
    prob_apriori = _config_.get_default_apriori_linkage_ratio()
    log_pos_posterior = 0
    log_neg_posterior = 0
    for v1, v2, a1_domain, a2_domain, a1_size, a2_size in all_datas:

        product_sizes = float(a1_size * a2_size)
        sum_sizes = float(a1_size + a2_size)

        neg_element, pos_element = _compute_probability_single_observation(a1_domain, a2_domain, product_sizes, sum_sizes,
                                                                           v1, v2, True)

        log_pos_posterior += log(max(pos_element, 1e-100))
        log_neg_posterior += log(max(neg_element, 1e-100))
    prob = compute_bayes_probability_from_log(log_neg_posterior, log_pos_posterior, prob_apriori)
    return prob


def compute_bayes_probability_from_log(log_neg_posterior, log_pos_posterior, prob_apriori_equivalence):
    """
    Compute bayes probability starting from priori, log of pos and neg posterior
    :param log_neg_posterior: 
    :param log_pos_posterior: 
    :param prob_apriori_equivalence: 
    :return: 
    """
    # now harmonize pos and neg and recompute original value
    diff = log_pos_posterior - log_neg_posterior
    if diff > 50:
        return 1
    elif diff < -50:
        return 0
    else:
        avg_posneg = (log_pos_posterior + log_neg_posterior) / 2
        log_pos_posterior -= avg_posneg
        log_neg_posterior -= avg_posneg
        pos_posterior = exp(log_pos_posterior)
        neg_posterior = exp(log_neg_posterior)
        result = pos_posterior * prob_apriori_equivalence / \
                 (pos_posterior * prob_apriori_equivalence + neg_posterior * (1 - prob_apriori_equivalence))
        return result

def compute_jaccard_similarity(v1, v2, containment = False):
    """
    Compute jaccard similarity between set of tokens.
    Numeric values goes double
    :param v1:
    :param v2:
    :param containment if True, use jaccard containment
    :return:
    """
    weight_token = lambda val: 0.5 + 0.5 * string_utils.is_token_numeric(val)
    denominator = min(sum(weight_token(x) for x in v1), sum(weight_token(x) for x in v2)) \
        if containment else sum(weight_token(x) for x in v1 | v2)
    return stats_utils.safe_divide(sum(weight_token(x) for x in v1 & v2), denominator)

