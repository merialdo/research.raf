import collections
import itertools

from tqdm import tqdm

import pipeline.cluster_utils
from adapter import coma_adapter
from config import constants
from config.bdsa_config import _config_
from model import dataset
from model.bdsa_data_transformed import BdsaDataTransformed
from model.clusters import SaClusterRules, InterSourceClusterRules, FreeClusterRules
from pipeline import pipeline_common, linkage_step, cluster_utils
from pipeline.linkage_step import compute_page_linkage_probability, debug_linkage
from pipeline.pipeline_abstract import AbstractPipeline
from pipeline.pipeline_common import ClusteringOutput, ITERATIONS_SIMILARITY
from pipeline.cluster_utils import WeightedEdge
from model.bdsa_data import BdsaData
from utils import prob_utils, bdsa_utils, io_utils

LINKAGE_ITER_SIMILARITY = 'linkage_iteration_similarity'

VALID_EDGES = 'VALID_EDGES'

NB_SELECTED_PAIRS = 'nb_pairs_source_attributes'

"""
This step computes the equivalence probability between selected pairs of attributes. 

"""


class PipelineComputeWeights(AbstractPipeline):

    def run(self, data):
        result = {}
        self.debug_stats = data[1]
        for cat, clustering_output in data[0].items():
            if self._use_coma:
                self._coma = coma_adapter.ComaAdapter(clustering_output.bdsa_data)
            result[cat] = self.compute_linkage_and_alignment(clustering_output, cat)

        return result, self.debug_stats

    def name(self):
        return "ComputeWeights"

    def need_input(self):
        return True

    def need_output(self):
        return True

    def __init__(self, _debug_matching_score=False, _use_coma=False):
        """

        :param do_linkage_iterations: if false, only schema alignment is launched. In some modes linkage iterations is done
        only on first step of alignment (and not after tagging)
        :param _use_coma use Coma++ instead of probabilistic match
        """

        self.nb_iterations = 1 if _config_.get_record_linkage_method() == _config_.RecordLinkageMethod.NONE \
            else _config_.get_number_of_iterations()

        # Cf. run
        self.debug_stats = None
        self.continue_iterations = True
        self.previous_edges_original = None
        self.debug_matching_score = _debug_matching_score
        self._use_coma = _use_coma
        self._coma = None

    def compute_linkage_and_alignment(self, clustering_output: ClusteringOutput, cat: str):
        """
        Compute all probability of match between attributes
        :return: triples (sa1, sa2, weight)
        """

        bdsa_data = clustering_output.bdsa_data
        source2page_candidate_pairs = linkage_step.initialize_linkage_data(bdsa_data, clustering_output)

        if LINKAGE_ITER_SIMILARITY not in self.debug_stats[cat] and _config_.get_record_linkage_method() \
                != _config_.RecordLinkageMethod.NONE:
            self.debug_stats[cat][LINKAGE_ITER_SIMILARITY] = []
        if ITERATIONS_SIMILARITY not in self.debug_stats[cat]:
            self.debug_stats[cat][ITERATIONS_SIMILARITY] = []
        if NB_SELECTED_PAIRS not in self.debug_stats[cat]:
            self.debug_stats[cat][NB_SELECTED_PAIRS] = []
        if VALID_EDGES not in self.debug_stats[cat]:
            self.debug_stats[cat][VALID_EDGES] = []

        # Do N iterations
        for i in range(self.nb_iterations):
            is_last_iteration = i == self.nb_iterations - 1
            is_first_iteration = i == 0
            self.do_schema_alignment(bdsa_data, clustering_output, is_last_iteration, cat)

            if not is_last_iteration and _config_.get_record_linkage_method() in \
                    [_config_.RecordLinkageMethod.PROB, _config_.RecordLinkageMethod.CLASS]:

                similarity = self.do_record_linkage(bdsa_data, clustering_output, is_first_iteration,
                                                    source2page_candidate_pairs)
                if similarity > 0:
                    self.debug_stats[cat][ITERATIONS_SIMILARITY].append\
                        ('Alignment - linkage evolution (step %d): %f ' % (i, similarity))
                    print('Similarity with previous family linkage: %f (if heuristically different then show 0)' % similarity)
                if similarity >= _config_.get_min_clustering_similarity_to_stop_iteration():
                    break

        if _config_.get_record_linkage_method() == _config_.RecordLinkageMethod.PROB:
            del source2page_candidate_pairs
            debug_linkage(self.debug_stats, bdsa_data, cat, clustering_output)

        self.continue_iterations = self._compute_new_edges_verify_equals(cat, clustering_output)

        if _config_.debug_mode():
            self.debug_stats[cat]['final iteration'] = pipeline_common.analyze_clustering_results(clustering_output)
        return clustering_output

    def _compute_new_edges_verify_equals(self, cat, clustering_output):
        """
        Compute edges of original attributes.
        If they are equivalent to previous step, then stop iterations.

        :param clustering_output:
        :return:
        """
        new_original_edges = []
        continue_iterations = True
        for cid, source2sas in clustering_output.sa_clusters.items():
            all_sas = sorted(sa for sa in set().union(*source2sas.values()) if not (sa.is_generated()))
            cluster_utils.add_group_edges(new_original_edges, [all_sas], 2)
        new_original_edges_set = set(new_original_edges)
        if self.previous_edges_original is not None:
            if self.previous_edges_original == new_original_edges_set:
                self.debug_stats[cat][pipeline_common.ITERATIONS_SIMILARITY].append('100% similarity')
                continue_iterations = False
            else:
                similarity_gross = 1 - len(new_original_edges_set - self.previous_edges_original) / len(self.previous_edges_original)
                self.debug_stats[cat][pipeline_common.ITERATIONS_SIMILARITY].append\
                    ('similarity (gross): %f' % (similarity_gross))
        self.previous_edges_original = new_original_edges_set
        return continue_iterations

    def do_record_linkage(self, bdsa_data, clustering_output, is_first_iteration, source2page_candidate_pairs):
        # If method is ADD, then existing pairs should NOT be evaluated (as they will be added anyway)
        if _config_.get_record_linkage_behavior() == _config_.RecordLinkageBehavior.ADD:
            current_pairs = frozenset(cluster_utils.build_all_pairs_generic(clustering_output.page_clusters.values()))
            source2page_candidate_pairs_new = collections.defaultdict(set)
            for source_pair, page_pairs in source2page_candidate_pairs.items():
                for page_pair in page_pairs:
                    if tuple(sorted(page_pair)) not in current_pairs:
                        source2page_candidate_pairs_new[source_pair].add(page_pair)
            source2page_candidate_pairs = source2page_candidate_pairs_new
        elif _config_.get_record_linkage_behavior() == _config_.RecordLinkageBehavior.DELETE:
            source2page_candidate_pairs = collections.defaultdict(set)
            for source2pages in clustering_output.page_clusters.values():
                for source_pair in itertools.combinations(sorted(source2pages.keys()), 2):
                    source2page_candidate_pairs[source_pair].update(
                        itertools.product(source2pages[source_pair[0]], source2pages[source_pair[1]]))

        if _config_.get_record_linkage_method() == _config_.RecordLinkageMethod.PROB:
            page_edges = compute_page_linkage_probability(source2page_candidate_pairs,
                                                          bdsa_data, clustering_output)

        if _config_.get_record_linkage_behavior() == _config_.RecordLinkageBehavior.ADD:
            page_edges.extend(WeightedEdge(p[0], p[1], 2) for p in current_pairs)
        similarity = pipeline.cluster_utils.partition_using_agglomerative(page_edges, clustering_output.page_clusters,
                                                                          FreeClusterRules() if _config_.do_allow_linkage_same_source() else InterSourceClusterRules(),
                                                                          _config_.get_min_elements_in_identical_clusters_to_stop_iteration())
        del page_edges
        return similarity

    def do_schema_alignment(self, bdsa_data, clustering_output, is_last_iteration:bool, cat:str):
        """
        Select candidate attribute pairs and do schema alignment
        :param bdsa_data: 
        :param clustering_output: 
        :param i:p
        :return: 
        """
        sa_edges = []
        if self.previous_edges_original:
            sa_edges.extend(self.previous_edges_original)
        clustering_output.sa_clusters.clear()
        clustering_output.att_matches.clear()
        source_pair2sa_pair2nb_value_matches = self.select_candidate_sa_pairs(bdsa_data, clustering_output)

        self.debug_stats[cat][NB_SELECTED_PAIRS].append(
            sum(len(sap2nb.keys()) for sap2nb in source_pair2sa_pair2nb_value_matches.values()))
        sa_edges.extend(self.compute_alignment_probabilities(source_pair2sa_pair2nb_value_matches,
                                                        clustering_output, is_last_iteration))

        if self.debug_matching_score:
            self._generate_results_debug_csv(bdsa_data, sa_edges)

        self.debug_stats[cat][VALID_EDGES].append(len(sa_edges))
        del source_pair2sa_pair2nb_value_matches
        pipeline.cluster_utils.partition_using_agglomerative(sa_edges, clustering_output.sa_clusters,
                                                             SaClusterRules(bdsa_data.sa2urls))
        pipeline_common.remove_non_master_attributes(clustering_output.sa_clusters)
        del sa_edges

    def _generate_results_debug_csv(self, bdsa_data, sa_edges):
        """
        Method used for debug, it outputs 1 csv
        """
        n = 1
        existing_pairs = dataset.Dataset(['id', 'source1', 'att1', 'source2', 'att2', 'score'])
        dir_debug = io_utils.build_directory_output('debug_%s' % _config_.get_category())
        for sa_edge in sa_edges:
            att_l = sa_edge.node1
            att_r = sa_edge.node2
            existing_pairs.add_row({'id': str(n), 'source1': att_l.source.site, 'att1': att_l.name,
                                    'source2': att_r.source.site, 'att2': att_r.name,
                                    'score': sa_edge.weight})
            # data_l = dataset.Dataset(['url', att_l.name])
            # data_r = dataset.Dataset(['url', att_r.name])
            # for pagel in bdsa_data.source2pages[att_l.source]:
            #     data_l.add_row({'url': pagel.url, att_l.name: bdsa_data.page2sa2value[pagel].get(att_l, '')})
            # for pager in bdsa_data.source2pages[att_r.source]:
            #    data_r.add_row({'url': pager.url, att_r.name: bdsa_data.page2sa2value[pager].get(att_r, '')})
            # data_l.export_to_csv(dir_debug, '%d_%s_%s' % (n, att_l.source.site, att_l.name), False)
            # data_r.export_to_csv(dir_debug, '%d_%s_%s' % (n, att_r.source.site, att_r.name), False)
            n += 1
        existing_pairs.export_to_csv(dir_debug,  'source_att_matching_results', False)

    def select_candidate_sa_pairs(self, bdsa_input: BdsaData, clustering_output: ClusteringOutput):
        """
        Find candidate pairs (that share at least 1 value) 
        :return: 
        """

        transform = bdsa_input.get_transformed_data()
        source_pair2sa_pair2distinct_value_matches = collections.defaultdict(bdsa_utils.dd2_int_generator)
        for source2pages in tqdm(clustering_output.page_clusters.values(), desc="Select attribute pair candidates..."):
            if len(source2pages) > 1:  # If an ID is affected to a single source, it is useless
                value2sas = collections.defaultdict(set)
                for source, pages in source2pages.items():
                    for page in pages:
                        for sa, value in transform.get_sa2value_for_page(page).items():
                            if not(transform.is_common_value(value)):
                                value2sas[value].add(sa)
                for value, sas in value2sas.items():
                    for sa1, sa2 in itertools.combinations(sas, 2):
                        if sa1.source != sa2.source:
                        #if len(bdsa_input.sa2urls[sa1] & bdsa_input.sa2urls[sa2]) == 0:
                            source_pair2sa_pair2distinct_value_matches[tuple(sorted((sa1.source, sa2.source)))][
                                tuple(sorted((sa1, sa2)))][value] += 1
        source_pair2sa_pair2distinct_value_matches_filter = {}
        for source_pair, sa_pair2distinct_value_matches in source_pair2sa_pair2distinct_value_matches.items():
            source_pair2sa_pair2distinct_value_matches_filter[source_pair] = \
                {sa_pair: distinct_value_matches for sa_pair, distinct_value_matches in sa_pair2distinct_value_matches.items() \
                    if sum(distinct_value_matches.values()) >= _config_.get_min_common_values_attributes()}
        return source_pair2sa_pair2distinct_value_matches_filter

    def compute_alignment_probabilities(self, source_pair2sa_pair2nb_value_matches,
                                         clustering_output: ClusteringOutput, is_last_iteration:bool):
        """
        Provided a list of candidate sa pairs, outputs a list of alignment probabilities
        :param source_pair2sa_pair2nb_value_matches: 
        :param is_last_iteration: if debug is activated, in last iteration we output the data of matches
        :return:
        """

        source2ids = clustering_output.source2pids
        output_comparisons = []
        pbar = tqdm(total=sum(len(sa_pairs.keys()) for sa_pairs in source_pair2sa_pair2nb_value_matches.values()),
                    desc='Computing match probability...')
        for source_pair, candidates2val2nb in source_pair2sa_pair2nb_value_matches.items():
            source_indexable_pair = tuple(source_pair)
            if self._use_coma:
                sas1 = set()
                sas2 = set()
                for sa_pair in candidates2val2nb.keys():
                    sas1.add(sa_pair[0])
                    sas2.add(sa_pair[1])
                results = self._coma.compare_sas_group(sas1, sas2, source_indexable_pair[0].site,
                                             source_indexable_pair[1].site, _config_.get_category())
                for res in results:
                    if res.weight > _config_.get_min_edge_weight():
                        output_comparisons.append(res)
                pbar.update(len(candidates2val2nb))
            else:
                common_ids = source2ids[source_indexable_pair[0]] & source2ids[source_indexable_pair[1]]
                for sa_pair, val2nb in candidates2val2nb.items():
                    pbar.update(1)
                    sa_indexable_pair = tuple(sa_pair)
                    sa1 = sa_indexable_pair[0]
                    sa2 = sa_indexable_pair[1]
                    #if self._is_valid_candidate(sa_pair[0], sa_pair[1], val2nb, transformed):
                    weight = self.compute_alignment_probability_attribute_pair(clustering_output, common_ids,
                                                                                   sa1, sa2, is_last_iteration)
                    if weight >= _config_.get_min_edge_weight():
                        output_comparisons.append(WeightedEdge(sa1, sa2, weight))
        pbar.close()
        return output_comparisons

    def compute_alignment_probability_attribute_pair(self, clustering_output: ClusteringOutput, common_ids,
                                                     sa1, sa2, is_last_iteration:bool):
        """
        Compute probability that 2 attributes are aligned
        :param clustering_output:
        :param common_ids: common IDS between the 2 respective sources
        :param sa_pair: the 2 attributes
        :return:
        """
        bdsa_data = clustering_output.bdsa_data
        transformer = bdsa_data.get_transformed_data()
        value_occurrences = transformer.nb_distinct_occurrences_attributes if _config_.use_idf() else None
        value_matches = []
        for pid in common_ids:
            values1 = self._get_values_of_sa_for_product(pid, sa1, clustering_output)
            values2 = self._get_values_of_sa_for_product(pid, sa2, clustering_output)
            if len(values1) == 0 or len(values2) == 0:
                continue
            elif len(values1) == 1 and len(values2) == 1:
                value_matches.append((values1[0], values2[0], 1))
            else:
                # If a product is associated to multiple specifications in at least one source, then we will have multiple attribute values
                # and we cannot directly build a pair of values to compare. We must find another solution:
                if _config_.one_value_per_product() == _config_.OneValuePerProductLegacy.yes:
                    # First solution: build a single pair of values, with most common value for attribute in specifications of that product for each source
                    top_value1 = bdsa_utils.most_common_deterministic(collections.Counter(values1), 1)[0][0]
                    top_value2 = bdsa_utils.most_common_deterministic(collections.Counter(values2), 1)[0][0]
                    value_matches.append((top_value1, top_value2, 1))
                else:
                    # Second solution: build ALL possible pairs of values with a cartesian product of values1 and values2. Each pair will have a weight of 1 / (values1*values2)
                    if _config_.one_value_per_product() == _config_.OneValuePerProductLegacy.no:
                        weight_internal = min(len(values1), len(values2)) / (len(values1) * len(values2))
                    else:
                        weight_internal = 1 / (len(values1) * len(values2))
                    value_matches.extend((x, y, weight_internal) for x, y in itertools.product(values1, values2))
        if len(value_matches) > 0:
            weight, prior = prob_utils.compute_attribute_equivalent_probability(value_matches, sa1.name.split(
                constants.GENERATED_ATTS_SEPARATOR)[0], sa2.name.split(constants.GENERATED_ATTS_SEPARATOR)[0],
                                                                         transformer.get_transformed_value2occs(sa1),
                                                                         transformer.get_transformed_value2occs(sa2),
                                                                         bdsa_data.sa2size[sa1],
                                                                         bdsa_data.sa2size[sa2],
                                                                         value_occurrences,
                                                                         len(bdsa_data.sa2size))
            if _config_.debug_mode():
                sa_sorted = sorted([sa1, sa2])
                value_matches_repr = ((' - '.join(v1), ' - '.join(v2)) for v1,v2, _ in value_matches)
                value_matches_synthesis = collections.Counter(value_matches_repr).most_common(3)
                clustering_output.att_matches[sa_sorted[0]][sa_sorted[1]] = \
                    {'matches': value_matches_synthesis, 'score': weight, 'prior':prior}
        else:
            weight = 0
        return weight

    def _is_valid_candidate(self, sa1, sa2, val2nb: dict, transformed: BdsaDataTransformed):
        """
        Defines if a given pair of attributes is a valid candidate for matching (one may apply further filtering).
        :param sa2: 
        :param counter: 
        :return: 
        """

        if len(val2nb) == 1:
            value = next(iter(val2nb.keys()))
            nb_atts = transformed.nb_distinct_occurrences_attributes(value)
            if transformed.is_common_value(value):
                return False
        return True

    def _get_values_of_sa_for_product(self, pid, sa1, clustering_output: ClusteringOutput):
        """
        Return the values provided by a given attribute for a provided product (there may be more than one 
        if a source has more pages for a single product.
        If there are none, we return a single NONE value.

        :param pid: 
        :param sa1: 
        :return: 
        """

        values = []
        transformer = clustering_output.bdsa_data.get_transformed_data()
        for page in clustering_output.page_clusters[pid][sa1.source]:
            sa2values = transformer.get_sa2value_for_page(page)
            if sa1 in sa2values:
                values.append(sa2values[sa1])
        # if len(values) == 0:
        #     values = [None]
        return values
