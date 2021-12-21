import collections
import itertools

from tqdm import tqdm

import pipeline.cluster_utils
from model import datamodel
from pipeline import pipeline_common
from pipeline.pipeline_abstract import AbstractPipeline
from pipeline.pipeline_common import ClusteringOutput

from model.bdsa_data import BdsaData
from utils import tagger_utils, stats_utils, bdsa_utils, string_utils
from config.bdsa_config import _config_

MAX_JACCARD_SIMILARITY_ATTRIBUTES = 0.8

NEW_VALUES = 'NEW VALUES'
ATOMIC_VALUES = 'ATOMIC VALUES'

ATT_TAG = 'att_tags'

VAL_TAG = 'val_tags'

## Pipeline step to tag attribute value with atomic values.
# Used to extract new attribute values

class PipelineTag(AbstractPipeline):

    def __init__(self):
        self.debug_stats = None

    def run(self, data):
        result = {}
        self.debug_stats = data[1]
        for cat, cluster_output in data[0].items():
            result[cat] = self.compute_attribute_tagging_and_extraction(cluster_output, cat)

        return result, self.debug_stats

    def name(self):
        return "PipelineTag"

    def need_input(self):
        return True

    def need_output(self):
        return True

    def compute_attribute_tagging_and_extraction(self, output: ClusteringOutput, cat: str) -> ClusteringOutput:
        """
        Identify in complex attributes subsequences of atomic attribute values, and extract them.
        E.g.: features='color black', color='black' ---> @extracted_att_features_color@ ='black'
        :param output:
        :param cat:
        :return:
        """

        # While we compute atomic and complex we keep generated attributes: there may be a former isolated att that matched
        # with many generated, it would be good to use its values in dictionary. At the same time, original attributes that belong
        # to a big cluster with many generated, are probably good, so it makes no sense to tag them
        atomic, complex_sas = identify_atomic_and_complex_attributes(output)

        cid2name = {cid: "%d_%s" % (cid, most_freq_name) for cid, most_freq_name in output.find_name_for_clusters().items()}
        value2cname = _build_value_dictionary_from_atomic_attributes(atomic, output, cid2name)
        if _config_.do_debug_tags():
            self._compare_old_tag_with_new_ones(value2cname, output, cat)
        else:
            output.bdsa_data.remove_all_generated()

            # Values extracted from same cluster in which this attribute belong should not be used for tagging
            sa2cname = {sa: cid2name[cid] for sa, cid in output.get_sa2cid().items()}
            attributes_generator = extract_subattributes_from_complex_attributes(complex_sas, value2cname, output.bdsa_data, sa2cname)
            custom_filter = lambda sa, bdsa: sum(1 / len(value2cname[value]) if value in value2cname else 0.5 for value
                                                in bdsa.sa2value2occs[sa]) >= _config_.get_min_distinct_values_score()
            output.bdsa_data.launch_attribute_generation(attributes_generator, custom_filter)
        return output

    def _compare_old_tag_with_new_ones(self, values2cname:dict, output, cat:str):
        # If no atomic values are present, i.e. this is the first extraction, then just add current values to debug
        if ATOMIC_VALUES in self.debug_stats[cat]:
            new_values = values2cname.keys() - self.debug_stats[cat][ATOMIC_VALUES]
            vals = ["%s[%s]" % (value, '/'.join(values2cname[value])) for value in new_values]
            self.debug_stats[cat][NEW_VALUES] = vals
        self.debug_stats[cat][ATOMIC_VALUES] = values2cname.keys()

def identify_atomic_and_complex_attributes(output: ClusteringOutput):
    """
    Separate attributes in atomic (from which the dictionary will be extracted) and complex (that will be tagged). One attribute
    can pertain to both elements
    :param output:
    :return:
    """

    # If we need to compute HEAD and TAIL:
    if _config_.get_attributes_to_tag() == _config_.AttributesToTag.TAIL_CLUSTERS \
        or _config_.get_extract_dict_values_from() == _config_.ExtractDictValuesFrom.HEAD_CLUSTERS:
        ht_clusters = {stats_utils.HEAD: {}, stats_utils.TAIL: {}}
        stats_utils.compute_head_tail(output.sa_clusters.items(), lambda cid2sources_entry: len(cid2sources_entry[1]),
                                      lambda cid2sources_entry, ht: ht_clusters[ht].update(
                                          {cid2sources_entry[0]: cid2sources_entry[1]}))

    # Choice of clusters from which extract dictionary data
    clusters_for_dictionary = ht_clusters[stats_utils.HEAD] \
        if _config_.get_extract_dict_values_from() == _config_.ExtractDictValuesFrom.HEAD_CLUSTERS else dict(output.sa_clusters)

    source2sas_to_tag = collections.defaultdict(set)
    for sa in output.sa_isolated:
        source2sas_to_tag[sa.source].add(sa)

    if _config_.get_attributes_to_tag() == _config_.AttributesToTag.ALL_ATTRIBUTES:
        clusters_to_tag = output.sa_clusters
    else:
        clusters_to_tag = ht_clusters[stats_utils.TAIL]

    for cid, source2sas in clusters_to_tag.items():
        for source, sas in source2sas.items():
            source2sas_to_tag[source].update(sas)

    return clusters_for_dictionary, source2sas_to_tag


def _build_value_dictionary_from_atomic_attributes(atomic_cids, output: ClusteringOutput, cid2name:dict):
    """
    Build a map with atomic values and corresponding clusters
    Exclude values present in too many clusters/atomic values.
    :param atomic_cids:
    :param output:
    :return:
    """
    value2cluster_name = collections.defaultdict(set)

    # Total nb of cluster and isolated values
    # If a generated attribute is isolated it should NOT count in nb of clusters, as it would augment irrealistically nb of clusters.
    nb_clusters_and_isolated = len(output.sa_clusters) + sum(1 for sa in output.sa_isolated if not sa.is_generated())
    # We keep only values present in at least get_min_att_in_cluster attributes of a cluster
    value_filter_apriori = lambda value: _config_.get_min_len_atomic_values() <= len(
        value) <= _config_.get_max_len_atomic_values() # or string_utils.is_token_numeric(value)
    # For each cluster we extract all values
    for cid in tqdm(atomic_cids, desc='Build dictionary...'):
        useful_values = _extract_significant_values_from_cluster(output.sa_clusters[cid], output.bdsa_data.sa2value2occs, value_filter_apriori)
        for value in useful_values:
            value2cluster_name[value].add(cid2name[cid])

    return {val: cnames for val, cnames in value2cluster_name.items()
            if len(cnames) / nb_clusters_and_isolated <= _config_.get_max_ratio_occs()}


def _extract_significant_values_from_cluster(cluster_source2sas, sa2val2occs, value_filter_apriori):
    """
    Extract all values in a given cluster that comply to a provided filter.
    Also keeps only values present in at least [get_min_att_in_cluster] attributes.

    :param cluster_source2sas: Source attributes in cluster grouped by source
    :param sa2val2occs: a map with values and occurrences of each source attribute
    :param value_filter_apriori: the filter (given just the string value)
    :return:
    """
    value2nb_attributes = collections.Counter()
    # TODO if only one atomic attribute, then ignore
    for sas in cluster_source2sas.values():
        for sa in sas:
            if not sa.is_generated():
                set_value_in_attribute = set(value for value in sa2val2occs[sa].keys() if value_filter_apriori(value))
                value2nb_attributes.update(set_value_in_attribute)
    useful_values = set(
        value for value, occs in value2nb_attributes.items() if occs >= _config_.get_min_att_in_cluster())
    return useful_values


def extract_subattributes_from_complex_attributes(complex_sas, value2cname: dict, bdsa_data: BdsaData, sa2cname:dict):
    """
    Look for atomic values inside complex values, and apply them

    :param complex_sas:
    :param value2cname:
    :param bdsa_data:
    :return:
    """
    tagger = tagger_utils.Tagger(value2cname, _function_extractor, _function_joiner,
                                 _config_.get_tag_every_combination(),
                                 _config_.get_min_ngram(), _config_.get_max_ngram())
    # Navigate through sources with at least 1 complex att



    for source, sas in tqdm(sorted(complex_sas.items()), desc='Tagging complex attributes'):
        sas = set(sa for sa in sas if not sa.is_generated())
        for page in bdsa_data.source2pages[source]:
            # Keep only page atts that are potentially complex
            for sa in sas & bdsa_data.page2sa2value[page].keys():
                value = bdsa_data.page2sa2value[page][sa]
                # Values extracted from same cluster in which this attribute belong should not be used for tagging
                excluded_clusters = frozenset({sa2cname[sa]} if sa in sa2cname else set())
                detected_snippets = tagger.tag(value,excluded_clusters)
                # If more than one snippets are extracted for a particular cluster, then:
                # - if some of values were also associated to other clusters we exclude them
                # - the remaining values are joined together.
                yield from _convert_snippets_to_virtual_attributes(bdsa_data, detected_snippets, page, sa)
        if _config_.add_extraction_data_in_output():
            _add_extraction_data(sas, bdsa_data, sas, tagger)


def _convert_snippets_to_virtual_attributes(bdsa_data, cname2snippets_original, page, sa):
    """
    Example: features == languages [italian], [french], main color [black], manufacturer [kodak] alt colors [red], [white].
    Snippets are grouped by cluster they come from)
    We have 2 snippets for languages, 1 for color, 1 for manufacturer and 3 for alt colors (BECAUSE ALSO black is extracted).
    We furtherly group snippet groups by their size : color, manufacturer --> language, altcolor.
    For each group we build virtual attributes for each snippet. If the do_assign_snippets_one_cluster is activated,
    then we do not create a v.a. for same snippet if it has already been assigned (e.g. black won't be assigned to alt colors).

    :param bdsa_data:
    :param cname2snippets_original:
    :param page:
    :param sa:
    :return:
    """
    virtual_instances_from_this_attribute_instance = []
    already_used_snippets = set()
    cname_snippets_copy = dict(cname2snippets_original)
    while len(cname_snippets_copy) > 0:
        # Now remove already used snippets
        if _config_.do_assign_snippets_one_cluster():
            cname_snippets_copy = {cname: set(snip for snip in snippets if snip not in already_used_snippets)
                                              for cname, snippets in cname_snippets_copy.items()}
        lower_snippet_size = min(len(snippets) for snippets in cname_snippets_copy.values())
        cname_snippet_this_size = {cname: snippets for cname, snippets in cname_snippets_copy.items() if len(snippets) == lower_snippet_size}
        for cname in cname_snippet_this_size:
            del cname_snippets_copy[cname]
        for cname, snippets in cname_snippet_this_size.items():
            if len(snippets) > 0:
                att_value = ' '.join(sorted(snippets))
                virtual_instances_from_this_attribute_instance.append(bdsa_data.GeneratedAttributeOccurrence(sa.name, att_value, page, cname))
                already_used_snippets.update(snippets)
    return virtual_instances_from_this_attribute_instance


def _add_extraction_data(att_names, bdsa_data, sas, tagger:tagger_utils.Tagger):
    """
    Highlights tagged substrings in original values
    :param att_names: names of non
    :param bdsa_data:
    :param sas: all complex attributes
    :param valuecache: cache with tagged elements
    :return:
    """
    for sa in sas:
        if sa.name in att_names:
            tops = bdsa_data.sa2topvalues[sa]
            for top_index in range(len(tops)):
                current_top_tuple = tops[top_index]
                snippets = tagger.tag(current_top_tuple[0])
                if snippets and len(snippets) > 0:
                    repres = ','.join(str(x) for x in snippets.items())
                    tops[top_index] = ("%s@@@%s" % (current_top_tuple[0],
                                                    repres), current_top_tuple[1])


## Help functions
def _function_extractor(gram: list, cids: set):
    """
    Used as parameter in tagger class.
    Snippet of text with a tag (or with nothing).
    :param gram:
    :param cids:
    :return: tuple with text and CID of cluster in which text was found, or none if no tag was extracted from text
    >>> _function_extractor(['3', 'cm'], {5})
    ('3 cm', {5})
    >>> _function_extractor(['color', 'black', 'compact'], {})
    ('color black compact', set())
    """
    return ' '.join(gram), cids if cids else set()


def _function_joiner(left: dict, tag: tuple, right: dict):
    """
    Used as parameter in tagger class.
    Joins the extracted snippets found in left and right part with the central one.
    :param left:
    :param tag:
    :param right:
    :return:
    >>> l = collections.defaultdict(set)
    >>> l.update({1:{'black'}, 2:{'black','white'}})
    >>> r = {2:{'red'}, 3:{'red'}, 4:{'16 GB'}}
    >>> res = _function_joiner(l, ('32 GB', {5}), r)
    >>> [(k, sorted(v)) for k, v in res.items()]
    [(1, ['black']), (2, ['black', 'red', 'white']), (3, ['red']), (4, ['16 GB']), (5, ['32 GB'])]
    """
    left = left or collections.defaultdict(set)
    if right:
        for cid, values in right.items():
            left[cid].update(values)

    if tag[1]:
        for cid in tag[1]:
            left[cid].add(tag[0])
    return left


if __name__ == "__main__":
    import doctest
    doctest.testmod()
