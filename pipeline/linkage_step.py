import collections
import copy
import itertools
import random

from tqdm import tqdm

from config.bdsa_config import _config_
from model import dataset
from model.bdsa_data import BdsaData
from model.clusters import FreeClusterRules, InterSourceClusterRules
from model.datamodel import SourceSpecifications
from pipeline import cluster_utils
from pipeline.cluster_utils import WeightedEdge
from pipeline.pipeline_common import ClusteringOutput
from utils import bdsa_utils, blocking_graph, prob_utils
from utils.blocking_graph import MetablockingGraph

LINKAGE_DIFFERENCES = 'linkage differences'


def initialize_linkage_data(bdsa_data, clustering_output):
    """
    Initial data for linkage. In particular, define initial linkage for first alignment step, AND candidate record
    linkage pairs (blocking).
    :param bdsa_data:
    :param clustering_output:
    :return:
    """

    # First page clustering is the inlput data
    if len(clustering_output.page_clusters) == 0 or _config_.do_restart_linkage():
        clustering_output.page_clusters = copy.deepcopy(bdsa_data.pid2source2pages)

    # If config is not probabilistic, then just use as initial clusters the input linkage and return
    if _config_.get_record_linkage_method() != _config_.RecordLinkageMethod.PROB:
        return None

    #If method is delete, it makes no sense to do candidate selection, as we must re-analyze EVERY existing pair.
    if _config_.get_record_linkage_behavior() == _config_.RecordLinkageBehavior.DELETE:
        return None

    rare_value2source2pages, unique_value2source2pages = _build_rare_and_unique_values_map(bdsa_data)
    ### INITIAL LINKAGE ###


    ### CANDIDATE PAIRS ###
    metablocking_graph_for_candidate_pairs = _build_metablocking_graph_linkage(bdsa_data, rare_value2source2pages,
                                                                               unique_value2source2pages)
    source2page_candidate_pairs = metablocking_graph_for_candidate_pairs.get_candidates()
    del metablocking_graph_for_candidate_pairs
    return source2page_candidate_pairs

def _build_source2aids(clustering_output):
    """
    Return list of AID for each source
    :param clustering_output:
    :return:
    """
    source2aids = collections.defaultdict(set)
    for aid in clustering_output.sa_clusters.keys():
        for source in clustering_output.sa_clusters[aid].keys():
            source2aids[source].add(aid)

    return source2aids


def select_candidate_page_pairs_class(source2aids, bdsa_data, clustering_output, feature_importance: dict):
    """
    Select pairs of pages that share k common values

    :param bdsa_data:
    :param clustering_output:
    :param important_aids:
    :param k:
    :return:
    """
    # Build inverted indexes
    aid2val2source2pages = collections.defaultdict(bdsa_utils.dd2_set_generator)
    for source, aids in tqdm(source2aids.items(),
                             desc='Build inverted index for candidate linkage pages...'):
        aid2attributes = {aid: clustering_output.sa_clusters[aid][source] for aid in aids
                          if aid in feature_importance.keys()}
        for page in bdsa_data.source2pages[source]:
            for aid, sas in aid2attributes.items():
                # should be only one
                for sa in sas:
                    sa2value = bdsa_data.get_transformed_data().get_sa2value_for_page(page)
                    if sa in sa2value:
                        aid2val2source2pages[aid][sa2value[sa]][page.source].add(page)

    page_pair_metablocking = blocking_graph.MetablockingGraph(_config_.get_min_blocking_score_linkage())
    pbar = tqdm(total=sum(len(source2pages) * (len(source2pages) - 1) / 2
                          for val2source2pages in aid2val2source2pages.values()
                          for source2pages in val2source2pages.values()),
                desc='Select candidate page pairs')
    for aid, val2source2pages in aid2val2source2pages.items():
        for source2pages in val2source2pages.values():
            for s1, s2 in itertools.combinations(source2pages.keys(), 2):
                pbar.update()
                for page1, page2 in itertools.product(source2pages[s1], source2pages[s2]):
                    page_pair_metablocking.increment_weigth(page1, page2, s1, s2, feature_importance[aid])
    pbar.close()

    return page_pair_metablocking.get_candidates()


def _build_value2transformed_page_for_source(source: SourceSpecifications, bdsa_data:BdsaData):
    """
    Build inverse map value2pages
    :param bdsa_data:
    :param pages:
    :return:
    """
    value2pages = collections.defaultdict(set)
    transformer = bdsa_data.get_transformed_data()
    for page in bdsa_data.source2pages[source]:
        for sa, value in transformer.get_sa2value_for_page(page).items():
            if transformer.is_value_non_isolated(value) \
                    and not(transformer.is_common_value(value)):
                value2pages[value].add(page)
    return value2pages


def _find_rare_and_unique_values_for_source(source_size, rare_value2source2pages, source, unique_value2source2pages,
                                            value2pages):
    """
    Detect unique and rare values and add them to maps.
    If 2 pages share enough rare/unique values, then we select them as candidate for blocking
    We start from rarer values to more frequent, we add values to rare map
    until both are true:
    * we passed all the pages of the source (i.e. there is at least 1 rare value per page)
    * we arrived at values with > 5 occurrences in pages (or source-1)
    OR until we have more values than [nb of pages * 2]

    Unique values are present in only 1 page of the source.

    :param pages: current pages of source
    :param rare_value2source2pages: map to fulfill
    :param source: current source
    :param unique_value2source2pages: map to fulfill
    :param value2pages:
    :return:
    """
    # Now try to include all pages in count
    pages_considered = set()
    count_sum = 0
    sorted_val2pages = sorted(value2pages.items(), key=lambda x: (len(x[1]), tuple(sorted(x[0]))))
    for value, pages in sorted_val2pages:
        count = len(pages)
        count_sum += count
        pages_considered.update(pages)
        rare_value2source2pages[value][source].update(pages)
        if count == 1:
            unique_value2source2pages[value][source].update(pages)
        # First condition: all pages are passed, and we used all values with  value <= 5. Second: not too much value
        if len(pages_considered) == source_size and count > min(source_size - 1, 4):
            break
        if count_sum > source_size * 2:
            break
        if count > 30:
            break
    del pages_considered


def compute_page_linkage_probability(source_pair2page_pair, bdsa_data: BdsaData,
                                     clustering_output: ClusteringOutput):
    """
    Provided a list of candidate page pairs, outputs a list of linkage probabilities
    :param: current_page_edges: existing edges of linked pages. If method is ADD, for these pairs we would not compute
    weight (as they would be added anyway). If method is DELETE, then weight is computed ONLY for these pairs (no
    other pair would be added).
    :param source_pair2page_pair:
    :return:
    """
    source2aids = _build_source2aids(clustering_output)

    transformed_data = bdsa_data.get_transformed_data()
    output_comparisons = []
    pbar = tqdm(total=sum(len(pages) for pages in source_pair2page_pair.values()), desc='Computing page linkage probability...')
    for source_pair, page_pairs in source_pair2page_pair.items():
        source1 = source_pair[0]
        source2 = source_pair[1]
        # Common attribute IDs (attribute in same cluster) between sources
        common_aids = source2aids[source1] & source2aids[source2]
        aid2sas1 = {}
        aid2sas2 = {}
        for aid in common_aids:
            # For each AID, which are the attribute pairs that we can compare (eg: 0 --> brand/brand, brand/manufacturer, brand name/manufacturer...)
            if _config_.do_exclude_generated_from_family_linkage():
                aid2sas1[aid] = [sa for sa in clustering_output.sa_clusters[aid][source1] if not sa.is_generated()]
                aid2sas2[aid] = [sa for sa in clustering_output.sa_clusters[aid][source2] if not sa.is_generated()]
            else:
                aid2sas1[aid] = clustering_output.sa_clusters[aid][source1]
                aid2sas2[aid] = clustering_output.sa_clusters[aid][source2]
        for page_pair in page_pairs:
            pbar.update(1)
            page1 = page_pair[0]
            page2 = page_pair[1]
            _compute_page_pairs_match(aid2sas1, aid2sas2, bdsa_data, common_aids, output_comparisons, page1,
                                              page2, transformed_data)
    pbar.close()
    return output_comparisons


def _compute_page_pairs_match(aid2sas1, aid2sas2, bdsa_data, common_aids, output_comparisons, page1, page2,
                              transformed_data):
    value_matches = []
    sas1_page = bdsa_data.page2sa2value[page1].keys()
    sas2_page = bdsa_data.page2sa2value[page2].keys()
    at_least_one_match = False
    for aid in common_aids:
        sas_aid_pg1 = sas1_page & aid2sas1[aid]
        sas_aid_pg2 = sas2_page & aid2sas2[aid]
        for sa1, sa2 in itertools.product(sas_aid_pg1, sas_aid_pg2):
            val1 = transformed_data.get_sa2value_for_page(page1).get(sa1, None)
            val2 = transformed_data.get_sa2value_for_page(page2).get(sa2, None)
            if val1 is not None and val2 is not None:
                if val1 == val2:
                    at_least_one_match = True
                value_matches.append(
                    [val1, val2, transformed_data.get_transformed_value2occs(sa1),
                     transformed_data.get_transformed_value2occs(sa2),
                     bdsa_data.sa2size[sa1], bdsa_data.sa2size[sa2]])
    if at_least_one_match:
        weight = prob_utils.compute_page_linkage_accuracy(value_matches)
        if weight >= _config_.get_min_edge_weight_linkage():
            output_comparisons.append(WeightedEdge(page1, page2, weight))


def _build_rare_and_unique_values_map(bdsa_data:BdsaData):
    # 2 maps: 1 for rare and uncommon values
    unique_value2source2pages = collections.defaultdict(bdsa_utils.dd_set_generator)
    rare_value2source2pages = collections.defaultdict(bdsa_utils.dd_set_generator)
    for source, pages in tqdm(sorted(bdsa_data.source2pages.items(), key=lambda s2p: str(s2p[0])),
                              desc='Compute candidate page pairs...'):
        value2pages = _build_value2transformed_page_for_source(source, bdsa_data)
        _find_rare_and_unique_values_for_source(len(pages), rare_value2source2pages, source,
                                                unique_value2source2pages, value2pages)
    del value2pages
    return rare_value2source2pages, unique_value2source2pages


def _build_metablocking_graph_linkage(bdsa_data, rare_value2source2pages, unique_value2source2pages):
    # TODO manage add and delete
    metablocking = MetablockingGraph(150)
    metablocking.add_full_clique(bdsa_data.pid2source2pages.values(), 150)
    for val, source2pages in rare_value2source2pages.items():
        score = int(round(1 / sum(len(pages) for pages in source2pages.values()) * 1000))
        metablocking.add_full_clique([source2pages], score)
    for val, source2pages in unique_value2source2pages.items():
        score = int(round(1 / sum(len(pages) for pages in source2pages.values()) * 1000))
        metablocking.add_full_clique([source2pages], score)
    return metablocking


##### CLASSIFIER #####

def _build_feature_vector_row(bdsa_data, clustering_output, common_aids, page1, page2,
                              is_linked: bool = None, fv: list = None):
    """
    Build a feature vector for random forest classifier (comparison of aligned attributes). If positive is not null,
    add the output class
    :param bdsa_data:
    :param clustering_output:
    :param common_aids:
    :param page1:
    :param page2:
    :param is_linked:
    :param fv: is provided, element is added to fv if it has at least 1 feature
    :return:
    """
    if is_linked is None:
        row = {}
    else:
        row = {IS_LINKED_LABEL: 1 if is_linked is True else 0}

    for aid in common_aids:
        # should be 0 or 1 element
        for a1, a2 in itertools.product(clustering_output.sa_clusters[aid][page1.source],
                                        clustering_output.sa_clusters[aid][page2.source]):
            val1 = bdsa_data.get_transformed_data().get_sa2value_for_page(page1).get(a1, None)
            val2 = bdsa_data.get_transformed_data().get_sa2value_for_page(page2).get(a2, None)
            if val1 is not None and val2 is not None:
                row[aid] = 1 if val1 == val2 else 0

    if fv is not None and len(row) > 1:
        fv.append(row)

    return row


def _build_sample_pages_outside_cluster(all_pages_source, pages_source_in_cluster):
    """
    Return a sample of pages from all_pages_source that are not in pages_source_in_cluster.
    Useful to build negative examples
    :param all_pages_source:
    :param pages_source_in_cluster:
    :return:
    """
    neg_sample_size = min(len(all_pages_source), _config_.get_neg_sample_linkage() // 2)
    neg_sample = random.sample(all_pages_source, neg_sample_size)
    neg_sample = [x for x in neg_sample if x not in pages_source_in_cluster]
    return neg_sample


def debug_linkage(debug_stats, bdsa_data, cat, clustering_output:ClusteringOutput):
    """
    Debug linkage results
    """
    linkage_diffs = cluster_utils.cluster_differences(bdsa_data.pid2source2pages, clustering_output.page_clusters)
    if LINKAGE_DIFFERENCES not in debug_stats[cat]:
        debug_stats[cat][LINKAGE_DIFFERENCES] = []
    debug_stats[cat][LINKAGE_DIFFERENCES].append({
        'original_pairs': linkage_diffs.original_pairs, 'added_pairs_until_now': linkage_diffs.added_pairs,
        'deleted_pairs_until_now': linkage_diffs.deleted_pairs})
    if _config_.debug_mode():
        rows = []
        iteration = len(clustering_output.page_matches)
        for added in linkage_diffs.sample_added:
            rows.extend(_convert_page_pair_to_row(added, clustering_output, bdsa_data, 'Added', iteration))
        for deleted in linkage_diffs.sample_deleted:
            rows.extend(_convert_page_pair_to_row(deleted, clustering_output,  bdsa_data, 'Deleted', iteration))
        clustering_output.page_matches.append(rows)

def _convert_page_pair_to_row(pair, clustering:ClusteringOutput, bdsa_data:BdsaData, info:str, iteration:int):
    """
    Convert a page pair to a debug row, with source, url and key-values
    """
    rows = []
    page1 = pair[0]
    page2 = pair[1]
    spec1 = bdsa_data.page2sa2value[page1]
    spec2 = bdsa_data.page2sa2value[page2]
    s2aids = _build_source2aids(clustering)
    row_base = {'source1': pair[0].source.site, 'source2': pair[1].source.site,
            'url1': pair[0].url, 'url2': pair[1].url, 'info':info, 'iteration':iteration}
    for aid in s2aids[page1.source] & s2aids[page2.source]:
        a1s = clustering.sa_clusters[aid][page1.source]
        a2s = clustering.sa_clusters[aid][page2.source]
        for attpair in itertools.product(a1s, a2s):
            a1 = attpair[0]
            a2 = attpair[1]
            row = row_base.copy()
            row.update({
                'type': 'pair', 'att1': a1.name, 'att2': a2.name,
                'val1': spec1.get(a1,'MISSING'), 'val2': spec2.get(a2, 'MISSING')
            })
            rows.append(row)
        spec1 = {k:v for k,v in spec1.items() if k not in a1s}
        spec2 = {k:v for k,v in spec2.items() if k not in a2s}
    row = row_base.copy()
    row.update({'type': 'remaining',
                'val1': '\n'.join('%s: %s' % (k.name, v) for k, v in
                                   spec1.items()),
                'val2': '\n'.join('%s: %s' % (k.name, v) for k, v in
                                   spec2.items()),
                })
    rows.append(row)

    return rows


IS_LINKED_LABEL = 'IsLinked'