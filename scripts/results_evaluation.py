import collections
import getopt
import itertools
import sys
from enum import Enum

from tqdm import tqdm

from adapter import adapter_factory
from adapter.output_and_gt_adapter import import_cluster_detail_csv, CLUSTER_ID_REAL, \
    build_il_clustering_from_gt, build_il_clustering_from_ikgpp, ClusterDetailOutput, InstanceLevelClustering
from config import constants
from config.bdsa_config import _config_
from model import dataset
from model.datamodel import BUCKET_SIZE_PROVS
from utils import experiment_utils, bdsa_utils, io_utils, stats_utils
from test.test_utils import tsa, tp
from utils.experiment_utils import EvaluationMetrics

SOURCE_SIZE = 'Source size'

TARGET_ATTRIBUTE_SIZE = 'TARGET_ATTRIBUTE_SIZE'

CLUSTER_ID = 'cluster_id'

F_1 = 'F1'
R = 'R'
P = 'P'

RP = 'RP'
CP = 'CP'
TP = 'TP'

RESULT = 'Result'
ENTITY_ID = 'ENTITY_ID'

GROUND_TRUTH_COMPARISON_FILE_PREFIX = 'ground_truth_comparison'
Evaluation = collections.namedtuple('Evaluation', 'precision recall tp rp cp')


class SchemaLevelEvaluation(Enum):
    NONE = 1  # No evaluation is made at schema-level, only instance-level
    WEIGHTED = 2  # Evaluation at schema-level is weighted (on size of smallest source-attribute)
    NON_WEIGHTED = 3  # Evaluation at schema-level is non-weighted

EvaluationResults = collections.namedtuple('EvaluationResults', 'schema instance_lib instance_cons')

class ResultsEvaluator:
    """
    Evaluation of BDSA alignment algorithm results against ground truth, both at schema-level and instance-level.
    """
    def __init__(self, schema_level_gt_path=None, instance_level_gt_path=None, category=None, pair_filter=None,
                 additional_columns_schema_gt=False):
        """

        :param schema_level_gt_path:
        :param instance_level_gt_path:
        :param category:
        :param pair_filter:
        :param additional_columns_schema_gt: retrieve all column data to schema GT
        """
        self.schema_gt_path = schema_level_gt_path or _config_.get_ground_truth_path()
        self.instance_gt_path = instance_level_gt_path or _config_.get_ground_truth_instance_level_path()
        self.cluster_detail_output = None
        self.category = category or _config_.get_category()
        self._schema_gt = None
        self._instance_gt = None
        self.pair_filter = pair_filter
        self.additional_columns_schema_gt = additional_columns_schema_gt

    def launch_evaluation_files(self, cdo_path, instance_level_output_path,
                          schema_level_evaluation=SchemaLevelEvaluation.WEIGHTED,
                          do_partitioned_evaluation=False, output_gt_comparison=False) -> EvaluationResults:
        """
        Same as launch_evaluation but provide output files instead of objects
        :param cdo_path:
        :param instance_level_output_path:
        :param schema_level_evaluation:
        :param do_partitioned_evaluation:
        :param output_gt_comparison:
        :return:
        """
        cluster_detail = import_cluster_detail_csv(CLUSTER_ID, cdo_path, extract_all_columns=True, category_input=self.category)
        ikgpp = None
        if instance_level_output_path:
            ikgpp = build_il_clustering_from_ikgpp(instance_level_output_path, self.category)
        return self.launch_evaluation(cluster_detail, ikgpp, schema_level_evaluation,
                                      do_partitioned_evaluation, output_gt_comparison)

    def compare_instance_level_results(self, ikgpp_path_1, ikgpp_path_2):
        print ('building ta2prov clustering')
        sas_elements = self.schema_gt.sa2clusters.keys()
        linkage = adapter_factory.linkage_factory()
        filter = lambda prov: prov.sa in sas_elements and len(linkage.ids_by_url(prov.url, prov.sa.source.site, self.category)) > 0
        prov2ta1 = build_il_clustering_from_ikgpp(ikgpp_path_1, self.category, filter)
        prov2ta2 = build_il_clustering_from_ikgpp(ikgpp_path_2, self.category, filter)
        ta2prov1 = bdsa_utils.multidict_inverter(prov2ta1.prov2ta)
        ta2prov2 = bdsa_utils.multidict_inverter(prov2ta2.prov2ta)

        clusters1 = set(tuple(sorted(provs)) for provs in ta2prov1.values())
        clusters2 = set(tuple(sorted(provs)) for provs in ta2prov2.values())
 #       clusters_only1 = clusters1 - clusters2
 #       clusters_only2 = clusters2 - clusters1
        pairs1 = set()
        pairs2 = set()
        for cluster in tqdm(clusters1, desc='Computing pairs from cluster1...'):
            pairs1.update(set(itertools.combinations(cluster, 2)))
        for cluster in tqdm(clusters2, desc='Computing pairs from cluster2...'):
            pairs2.update(set(itertools.combinations(cluster, 2)))
        #pairs1 = set().union(*(set(itertools.combinations(cluster, 2)) for cluster in clusters1))
        #pairs2 = set().union(*(set(itertools.combinations(cluster, 2)) for cluster in clusters2))
        print('End pairs. Build samples....')
        pairs_only1 = pairs1 - pairs2
        pairs_only2 = pairs2 - pairs1
        correct_fx = lambda pair: len(self.instance_gt.prov2ta[pair[0]] & self.instance_gt.prov2ta[pair[1]]) > 0
        pairs_correct_1 = set(pair for pair in pairs_only1 if correct_fx(pair))
        pairs_wrong_1 = pairs_only1 - pairs_correct_1
        pairs_correct_2 = set(pair for pair in pairs_only2 if correct_fx(pair))
        pairs_wrong_2 = pairs_only2 - pairs_correct_2

        print('Total pairs: 1 %d, 2: %d. Pairs only in 1 correct: %d, wrong: %d, only in 2 correct: %d, wrong %d'
              % (len(pairs1), len(pairs2), len(pairs_correct_1),len(pairs_wrong_1), len(pairs_correct_2), len(pairs_wrong_2)))
        sample_correct1 = stats_utils.safe_sample(pairs_correct_1, 150)
        sample_wrong1 = stats_utils.safe_sample(pairs_wrong_1, 150)
        sample_correct2 = stats_utils.safe_sample(pairs_correct_2, 150)
        sample_wrong2 = stats_utils.safe_sample(pairs_wrong_2, 150)
        print('printing...')
        ds = dataset.Dataset(['source1', 'source2', 'name1', 'name2', 'value1',
                              'value2', 'url1', 'url2', 'status', 'gt'])
        for pair in sample_correct1:
            ds.add_row(self._convert_provenance_to_row('Only first', pair, 'CORRECT'))
        for pair in sample_wrong1:
            ds.add_row(self._convert_provenance_to_row('Only first', pair, 'WRONG'))
        for pair in sample_correct2:
            ds.add_row(self._convert_provenance_to_row('Only second', pair, 'CORRECT'))
        for pair in sample_wrong2:
            ds.add_row(self._convert_provenance_to_row('Only second', pair, 'WRONG'))
        ds.export_to_csv(_config_.get_output_dir(), 'comparison_results', True)

    def _convert_provenance_to_row(self, status, pair, correctness):
        return {'source1': pair[0].sa.source.site, 'source2': pair[1].sa.source.site,
                'name1': pair[0].sa.name, 'name2': pair[1].sa.name,
                'value1': pair[0].value, 'value2': pair[1].value,
                'url1': pair[0].url, 'url2': pair[1].url,
                'status': status, 'gt': correctness}

    def launch_evaluation(self, cdo:ClusterDetailOutput, instance_level_output:InstanceLevelClustering=None,
                          schema_level_evaluation=SchemaLevelEvaluation.WEIGHTED,
                          do_partitioned_evaluation=False, output_gt_comparison=False) -> EvaluationResults:
        """
        Launch evaluation against GT: schema and/or instance-level evaluation and/or ground truth comparison file,
        according to parameters.
        :param cdo: cluster detail file. Mandatory even if only instance-level is needed (provide useful details about SAs)
        :param schema_level_evaluation: type of evaluation on schema-level: none, weighted, non-weighted.
        :param do_partitioned_evaluation: Add evaluation on partitioned data
        :param output_gt_comparison: Output file with a comparison between output and comparison GT
        :param instance_level_output: The output file for instance-level (if not provided, won't do instance-level analysis)
        :param filter: potential filter for pairs
        :return:
        """

        if output_gt_comparison:
            print('Building detailed comparison output...')
            results_ds = _build_detailed_comparison_with_ground_truth(self.schema_gt.sa2clusters, cdo)
            results_ds.export_to_csv(_config_.get_output_dir(), GROUND_TRUTH_COMPARISON_FILE_PREFIX, True)

        evaluation_schema_level = None
        if schema_level_evaluation != SchemaLevelEvaluation.NONE:
            print('Evaluating schema-level ...')
            weight_function = (lambda pair: _get_pair_weight(pair, cdo.sa2other_data)) \
                if schema_level_evaluation == SchemaLevelEvaluation.WEIGHTED else lambda sa_pair: 1
            evaluation_schema_level = _compute_precision_recall_sl(self.schema_gt.sa2clusters, cdo.sa2clusters,
                                                                   pair_weight_compute=weight_function,
                                                                   pair_filter=self.pair_filter)

        evaluation_instance_level_lib = None
        evaluation_instance_level_cons = None
        if instance_level_output is not None:
            print('Evaluating instance-level...')
            # Instance-level output should be evaluated on ALL attributes in ground truth.
            # However, there may be some experiments in which we launch algorithm in a subset of dataset (typically a
            # subset of sources). For this reason, we limit the evaluation on the union of GT sas AND sas in output.
            # ALL SAS managed by the algorithm are in cdo.sa2clusters.keys(), even the isolated ones.
            # Notice that we cannot use the IKGPP because it may miss some attribute occurrences if they are considered
            # as isolated.
            sas_to_evaluate = cdo.sa2clusters.keys() & self.schema_gt.sa2clusters.keys()
            evaluation_instance_level_lib, evaluation_instance_level_cons = \
                _compute_precision_recall_il(instance_level_output.prov2ta, self.instance_gt,
                                             sas_to_evaluate, cdo.sa2other_data, self.schema_gt.sa2clusters,
                                             do_partitioned_evaluation)
        return EvaluationResults(evaluation_schema_level, evaluation_instance_level_lib, evaluation_instance_level_cons)

    @property
    def schema_gt(self):
        if self._schema_gt is None:
            self._schema_gt = import_cluster_detail_csv(CLUSTER_ID_REAL, self.schema_gt_path, category_input=self.category,
                                                        extract_all_columns=self.additional_columns_schema_gt)
        return self._schema_gt

    @property
    def instance_gt(self):
        if self._instance_gt is None:
            il_gt = dataset.import_csv(self.instance_gt_path)
            self._instance_gt = build_il_clustering_from_gt(il_gt, self.category)
        return self._instance_gt

def _get_repartitions(sa2data:dict, sas_to_evaluate, sa2expected_cluster, il_ground_truth) -> tuple:
    """
    Generator that returns different repartition functions used to partition the dataset
    :param sa2data: information associated to each source attribute
    :param sas_to_evaluate: all SAS we must evaluate, useful to comput H/T
    :return: yields tuples name - repartition function
    """

    yield  (constants.HETEROGENEITY, 
	lambda sa: 'heterogeneous' if sa in il_ground_truth.heterogeneous_sa else 'homogeneous')
    # Head-tail attributes according to cardinality
    if _config_.do_experiments_attribute_cardinality():
        yield (constants.CARDINALITY,  _build_splitter_ht(
            sas_to_evaluate, lambda sa: int(sa2data[sa.get_original_attribute()][constants.CARDINALITY])))
    # Head-tail attributes according to attribute occurrences
    if _config_.do_experiments_attribute_size():
        head_tail_attributes_function = _build_splitter_ht(sas_to_evaluate,
                                lambda sa: int(sa2data[sa.get_original_attribute()][constants.OCCURRENCES]))
        yield (constants.OCCURRENCES, head_tail_attributes_function)
    # Head-tail TARGET predicates, i.e. clusters present in many sources
    if _config_.do_experiments_cluster_distinct_sources():
        # In this particular case HEAD-TAIL is computed on the attributes in GROUND TRUTH (as we don't know a-priori which
        # are the 'target attributes' and their size), while usually we compute it on whole dataset.

        cluster2sources = bdsa_utils.build_dict(sa2expected_cluster.items(), key_getter=lambda keyval: keyval[1],
                                                value_getter=lambda keyval: keyval[0].source, key_getter_returns_multi=True)
        head_tail_predicates_function = _build_splitter_ht(sas_to_evaluate,
                                lambda sa: max((len(cluster2sources[cid]) for cid in sa2expected_cluster[sa]),
                                               default=0))
        yield (TARGET_ATTRIBUTE_SIZE, head_tail_predicates_function)

    # If we want a quadrangle predicates vs attributes h-t
    if _config_.do_experiments_source_size() and _config_.do_experiments_cluster_distinct_sources():
        yield ('HT_ATTRIBUTES_PREDICATES',
               lambda sa: 'PRED_%s_ATT_%s' % (head_tail_predicates_function(sa), head_tail_attributes_function(sa)))

    if _config_.do_experiments_source_size():
        source2size = collections.Counter(sa.source for sa in sa2data.keys())
        yield (SOURCE_SIZE, _build_splitter_ht(sas_to_evaluate, lambda sa: source2size[sa.source]))

    # Notice that here we do not use head-tail (as we do not have a global count) but only top-bottom 50%
#    if _config_.do_experiments_source_linkage():
#        ordered_sources = adapter_factory.spec_factory().source_names_ordered_linkage_decreasing()
#        top_sources = ordered_sources[:round(len(ordered_sources) / 2)]
#        yield ('Sources with most linkage', lambda sa: 'top half' if sa in top_sources else 'bottom half')
    if _config_.do_experiments_attribute_linkage():
        yield (constants.LINKED_PAGES, _build_splitter_ht(
            sas_to_evaluate, lambda key:
            int(sa2data[key.get_original_attribute()][constants.LINKED_PAGES])))


def _build_splitter_ht(elements, size_getter):
    """
    Build a function that, given an element of dataset, return 'HEAD' or 'TAIL' if that element is H-T according to a
    certain metrics.
    :param elements:
    :param size_getter:
    :return:
    """
    head_elements = set()
    stats_utils.compute_head_tail(elements, lambda sa: size_getter(sa), lambda sa, ht: head_elements.add(sa) if ht == stats_utils.HEAD else 1)
    return lambda sa: stats_utils.HEAD if sa in head_elements else stats_utils.TAIL


def _compute_precision_recall_sl(sa2real_clusters:dict, sa2computed_clusters:dict,
                                 pair_weight_compute=lambda sa_pair:1, pair_filter=None) -> experiment_utils.EvaluationMetrics:
    """
    Computes precision and recall using pairs of attributes --> (1=same_cluster, 0=not_same_cluster) as input.
    Can provide results on particular partition of dataset

    :param sa2real_clusters:
    :param sa2computed_clusters:
    :param do_compute_weighted_evaluation: if True, positive pairs are weighted according to min nb of occurrences
    :return:
    >>> sa2real = {tsa('a'): {1, 5}, tsa('b'): {1, 5}, tsa('c'): {1}, tsa('d'): {2}, tsa('e'): {2}, tsa('f'): {2},\
    tsa('g'): {1,2}, tsa('h'):{324}, tsa('kjk'):{100}, tsa('fdfsfd'):{100}}
    >>> sa2comp = {tsa('a'): {1}, tsa('b'): {1}, tsa('c'): {1}, tsa('d'): {2}, tsa('e'): {2}, tsa('f'): {2},\
    tsa('h'): {1},  tsa('g'):{345} , tsa('ffff'):{1}}
    >>> order_splitter = lambda x: 'AD' if 'a' <= x.name <= 'd' else 'E+'
    >>> vowel_splitter = lambda x: 'vowel' if x.name in ['a', 'e'] else 'consonant'
    >>> repart2function = {'type': vowel_splitter, 'order': order_splitter}
    >>> global_evaluation, evaluation = _compute_precision_recall_sl(sa2real, sa2comp, repartition2function_splitter=repart2function)
    >>> global_evaluation.compact(), {name: eval.compact() for name, eval in evaluation.items()}
    (9, 12, 6), {'type__consonant': (2, 6, 4), 'type__vowel': (0, 0, 0), 'order__AD': (1, 1, 1), 'order__E+': (3, 7, 4)}
    """
    common_sas = sa2real_clusters.keys() & sa2computed_clusters.keys()
    computed_cid2sas = _invert_sa2cid(sa2computed_clusters, common_sas)
    expected_cid2sas = _invert_sa2cid(sa2real_clusters, common_sas)
    computed_pairs = _build_all_pairs(computed_cid2sas, pair_filter=pair_filter)
    expected_pairs = _build_all_pairs(expected_cid2sas, pair_filter=pair_filter)
    evaluation = build_precision_recall_from_computed_expected_positives(computed_pairs,
                                                                         expected_pairs, pair_weight_compute)
    return evaluation

def _build_detailed_comparison_with_ground_truth\
                (sa2expected_cluster, cdo:ClusterDetailOutput,
                 suffix_additional_cids='.1'):
    """
    Returns a comparison between the alignment results AND the ground truth.
    Considers only attributes both in ground truth & output, compares real and ocmputed ID
    :param sa2comp_cluster:
    :param sa2real_cluster:
    :return:
    >>> sa2comp_cluster = {tsa('AB'): {1}, tsa('ABCD'): {1,3}, tsa('A'): {1}, tsa('A+'): {1}, tsa('B'): {2}, tsa('B+'): {2}, \
                    tsa('C'): {3}, tsa('C+'): {3}, tsa('D'): {4}, tsa('D+'): {4}}
    >>> sa2expected_cluster = {tsa('AB'): {10,20}, tsa('ABCD'): {10,20,40}, tsa('A'): {10}, tsa('A+'): {10}, tsa('B'): {20}, tsa('B+'): {20}, \
                tsa('C'): {30}, tsa('C+'): {30}, tsa('D'): {40}, tsa('D+'): {40}}
    >>> res = _build_detailed_comparison_with_ground_truth(sa2expected_cluster, sa2comp_cluster)
    >>> sorted(res.rows, key=lambda x: x['name'])
    [{'source': 'test_site', 'name': 'A', 'cluster_id': 1, 'cluster_id_real': 10}, {'source': 'test_site', 'name': 'A+', 'cluster_id': 1, 'cluster_id_real': 10}, {'source': 'test_site', 'name': 'AB', 'cluster_id': 1, 'cluster_id_real': 10}, {'source': 'test_site', 'name': 'AB', 'cluster_id': -1, 'cluster_id_real': 20}, {'source': 'test_site', 'name': 'ABCD', 'cluster_id': 1, 'cluster_id_real': 10}, {'source': 'test_site', 'name': 'ABCD', 'cluster_id': 3, 'cluster_id_real': 40}, {'source': 'test_site', 'name': 'ABCD', 'cluster_id': -1, 'cluster_id_real': 20}, {'source': 'test_site', 'name': 'B', 'cluster_id': 2, 'cluster_id_real': 20}, {'source': 'test_site', 'name': 'B+', 'cluster_id': 2, 'cluster_id_real': 20}, {'source': 'test_site', 'name': 'C', 'cluster_id': 3, 'cluster_id_real': 30}, {'source': 'test_site', 'name': 'C+', 'cluster_id': 3, 'cluster_id_real': 30}, {'source': 'test_site', 'name': 'D', 'cluster_id': 4, 'cluster_id_real': 40}, {'source': 'test_site', 'name': 'D+', 'cluster_id': 4, 'cluster_id_real': 40}]
    >>> input_data = [(row['cluster_id_real'], row['cluster_id']) for row in res.rows]
    >>> _compute_precision(input_data, True)
    (9, 11, 16)
    """
    results_ds = dataset.Dataset(['source', 'name', CLUSTER_ID_REAL, CLUSTER_ID])  # This is the output
    expected_cluster2sa = bdsa_utils.multidict_inverter(sa2expected_cluster)
    comp_cluster2sa = bdsa_utils.multidict_inverter(cdo.sa2clusters)
    # For each attribute, we add one or more rows according to the number of clusters it has been assigned
    for sa in sa2expected_cluster.keys() & cdo.sa2clusters.keys():

        cids = set(cdo.sa2clusters[sa])
        real_cids = set(sa2expected_cluster[sa])

        # In general each attribute may pertain to N real clusters and M computed clusters.
        # For a given row, we show the IDs of most similar pair of clusters between the real and the
        # computed (IE the pair with most common SAS.
        while len(real_cids) > 0 and len(cids) > 0:
            candidates = itertools.product(cids, real_cids)
            cid, real_cid = max(candidates, key=lambda ccrc: len(comp_cluster2sa[ccrc[0]]
                                                                 & expected_cluster2sa[ccrc[1]]))
            results_ds.add_row(_sa2row(cid, real_cid, sa, cdo.sa2other_data))
            cids.remove(cid)
            real_cids.remove(real_cid)
        # We finally add remaining IDs, using last remained CID with a 0.1 to distinguish
        for cid in cids:
            results_ds.add_row(_sa2row(cid, real_cid+suffix_additional_cids, sa, cdo.sa2other_data))
        for real_cid in real_cids:
            results_ds.add_row(_sa2row(cid+suffix_additional_cids, real_cid, sa, cdo.sa2other_data))
    return results_ds

def _compute_precision_recall_il(prov_computed2ta, il_ground_truth, sas_to_evaluate, sa2data, sa2exp_cluster,
                                 do_partitioned_evaluation) -> EvaluationMetrics:
    """
    Compute evaluation given instance-based clusters
    :param ta2name_urls:
    :param ta2name_urls_gt:
    :return:
    >>> ta2nu_gt = {tp('dimensions', 'ebay','1'): {'l','w', 'h'}, tp('dimensions', 'ebay','2'): {'l','w'}, \
                    tp('dimensions', 'amz','1'): {'l','w', 'h'}}
    >>> ta2nu_out_correct = {tp('dimensions', 'ebay','1'): {'l','w', 'h'}, tp('dimensions', 'ebay','2'): {'l','w'}, \
                    tp('dimensions', 'amz','1'): {'l','w', 'h'}}
    >>> ta2nu_out_all = {tp('dimensions', 'ebay','1'): {'dim'}, tp('dimensions', 'ebay','2'): {'dim'}, \
                    tp('dimensions', 'amz','1'): {'dim'}}
    >>> ta2nu_out_only_lw = {tp('dimensions', 'ebay','1'): {'1'}, tp('dimensions', 'ebay','2'): set(), \
                    tp('dimensions', 'amz','1'): {'1'}}
    >>> sas_to_evaluate = set(prov.sa for prov in ta2nu_gt.keys())
    >>> res_lib, res_cons = _compute_precision_recall_il(ta2nu_out_correct, ta2nu_gt, sas_to_evaluate, None,None, False)
    >>> (res_lib.compact(), res_cons.compact())
    ((3, 3, 3), (1, 1, 1))
    >>> res_lib, res_cons = _compute_precision_recall_il(ta2nu_out_all, ta2nu_gt, sas_to_evaluate, None,None, False)
    >>> (res_lib.compact(), res_cons.compact())
    ((3, 3, 3), (1, 1, 3))
    >>> res_lib, res_cons = _compute_precision_recall_il(ta2nu_out_only_lw, ta2nu_gt, sas_to_evaluate, None,None, False)
    >>> (res_lib.compact(), res_cons.compact())
    ((1, 3, 1), (1, 1, 1))
    """
    # Get expected and computed results. Expected are limited to those in sas_to_evaluate, computed are limited to
    # provs in expected.
    prov_gt2ta = il_ground_truth.prov2ta
    prov2ta_gt_reduced = {prov: ta for prov, ta in prov_gt2ta.items() if prov.sa in sas_to_evaluate}
    prov2ta_computed_reduced = {prov: ta for prov, ta in prov_computed2ta.items() if prov in prov2ta_gt_reduced.keys()}

    metrics_liberal = _compute_liberal_il_metrics(do_partitioned_evaluation, il_ground_truth, prov2ta_computed_reduced,
                                                  prov2ta_gt_reduced, sa2data, sa2exp_cluster, sas_to_evaluate)
    metrics_conservative = EvaluationMetrics(1,1,1)#_compute_conservative_il_metrics(do_partitioned_evaluation, il_ground_truth,
                                                    #        prov2ta_computed_reduced, prov2ta_gt_reduced, sa2data,
                                                     #       sa2exp_cluster, sas_to_evaluate)
    return metrics_liberal, metrics_conservative


def _compute_conservative_il_metrics(do_partitioned_evaluation, il_ground_truth, prov2ta_computed_reduced,
                                     prov2ta_gt_reduced, sa2data, sa2exp_cluster, sas_to_evaluate):
    # Conservative GT: attribute 'match' if they share exactly the same TA
    ta_group2prov_gt = bdsa_utils.build_dict(prov2ta_gt_reduced.items(),
                                             key_getter=lambda prov2ta: tuple(sorted(prov2ta[1])),
                                             value_getter=lambda prov2ta: prov2ta[0])
    ta_group2prov_computed = bdsa_utils.build_dict(prov2ta_computed_reduced.items(),
                                                   key_getter=lambda prov2ta: tuple(sorted(prov2ta[1])),
                                                   value_getter=lambda prov2ta: prov2ta[0])
    metrics_conservative = _compute_metrics_il(ta_group2prov_computed, ta_group2prov_gt,
                                               do_partitioned_evaluation, sa2data, sa2exp_cluster,
                                               sas_to_evaluate, il_ground_truth)
    return metrics_conservative


def _compute_liberal_il_metrics(do_partitioned_evaluation, il_ground_truth, prov2ta_computed_reduced,
                                prov2ta_gt_reduced, sa2data, sa2exp_cluster, sas_to_evaluate):
    # Liberal Metrics: attribute 'match' if they share at least 1 TA
    ta2prov_gt = bdsa_utils.multidict_inverter(prov2ta_gt_reduced)
    ta2prov_computed = bdsa_utils.multidict_inverter(prov2ta_computed_reduced)
    metrics_liberal = _compute_metrics_il(ta2prov_computed, ta2prov_gt, do_partitioned_evaluation,
                                          sa2data, sa2exp_cluster, sas_to_evaluate, il_ground_truth)
    return metrics_liberal


def _compute_metrics_il(ta2prov_computed, ta2prov_gt, do_partitioned_evaluation, sa2data, sa2exp_cluster,
                        sas_to_evaluate, il_ground_truth):
    """
    Compute metrics instance-level, providing expected and computed clusters.
    Common method for liberal and conservative P-R

    :param computed_clusters: computed clusters of provs
    :param expected_clusters: expected clusters of provs
    :param do_partitioned_evaluation:
    :param sa2data: additional data associated to source attributes (useful for partitioning)
    :param sa2exp_cluster: GT clusters of source attributes (useful for partitioning)
    :param sas_to_evaluate: all SAS that must be evaluated (for part
    :return:
    """

    # In order to avoid problems of memory, we divide provs in buckets

    pbar = tqdm(total=BUCKET_SIZE_PROVS * (BUCKET_SIZE_PROVS + 1) / 2,
                desc='Compute metrics I.L.')
    total_expected = 0
    total_computed = 0
    total_true = 0
    if do_partitioned_evaluation:
        partitioned_fp = collections.defaultdict(bdsa_utils.counter_generator)
        partitioned_fn = collections.defaultdict(bdsa_utils.counter_generator)
        partitioned_tp = collections.defaultdict(bdsa_utils.counter_generator)
    for buck_i in range(BUCKET_SIZE_PROVS):
        for buck_j in range(buck_i, BUCKET_SIZE_PROVS):
            pbar.update(1)
            pairs_computed = _build_all_pairs(ta2prov_computed, filter_element=lambda el: el.bucket in [buck_i, buck_j],
                             pair_filter=lambda pair: (pair[0].bucket == buck_i and pair[1].bucket == buck_j) or
                                                      (pair[0].bucket == buck_j and pair[1].bucket == buck_i))
            pairs_expected = _build_all_pairs(ta2prov_gt, filter_element=lambda el: el.bucket in [buck_i, buck_j],
                             pair_filter=lambda pair: (pair[0].bucket == buck_i and pair[1].bucket == buck_j) or
                                                      (pair[0].bucket == buck_j and pair[1].bucket == buck_i))
            pairs_true = pairs_computed & pairs_expected
            total_computed += len(pairs_computed)
            total_expected += len(pairs_expected)
            total_true += len(pairs_true)
            if do_partitioned_evaluation:
                repartition_name = _compute_partition_bucket(il_ground_truth, pairs_computed, pairs_expected,
                                                             pairs_true, partitioned_fn, partitioned_fp, partitioned_tp,
                                                             sa2data, sa2exp_cluster, sas_to_evaluate)
    pbar.close()
    evaluation = EvaluationMetrics(total_true, total_expected, total_computed)
    if do_partitioned_evaluation:
        for group in partitioned_tp.keys() | partitioned_fp.keys() | partitioned_fn.keys():
            eval = experiment_utils.evaluation_metrics_with_falses(sum(partitioned_tp[group].values()),
                                                                           sum(partitioned_fp[group].values()),
                                                                           sum(partitioned_fn[group].values()))
            evaluation.add_subset("%s__%s" % (repartition_name, group), eval)
    return evaluation


def _compute_partition_bucket(il_ground_truth, pairs_computed, pairs_expected, pairs_true, partitioned_fn,
                              partitioned_fp, partitioned_tp, sa2data, sa2exp_cluster, sas_to_evaluate):
    false_positives = pairs_computed - pairs_true
    false_negatives = pairs_expected - pairs_true
    # Partitioned evaluation is always based on source attribute of a provenance. To avoid too many iterations,
    # we build a map (sa_pairs --> nb of provenances) and compute repartition on this.
    tp_grouped = collections.Counter(frozenset([prov[0].sa, prov[1].sa]) for prov in pairs_true)
    fp_grouped = collections.Counter(frozenset([prov[0].sa, prov[1].sa]) for prov in false_positives)
    fn_grouped = collections.Counter(frozenset([prov[0].sa, prov[1].sa]) for prov in false_negatives)
    for repartition_name, splitter in tqdm(_get_repartitions(sa2data, sas_to_evaluate, sa2exp_cluster, il_ground_truth),
                                           desc='Computing partitions...'):
        tp_partitions = _split_instances(splitter, tp_grouped)
        for group, cnt in tp_partitions.items():
            partitioned_tp[group].update(cnt)
        fp_partitions = _split_instances(splitter, fp_grouped)
        for group, cnt in fp_partitions.items():
            partitioned_fp[group].update(cnt)
        fn_partitions = _split_instances(splitter, fn_grouped)
        for group, cnt in fn_partitions.items():
            partitioned_fn[group].update(cnt)
    return repartition_name

def _split_instances(splitter, instances):
    """
    Split instance counter according to a splitter

    :param splitter:
    :param instances:
    :param instance_partitioned:
    :return:
    """
    instance_partitioned = collections.defaultdict(bdsa_utils.counter_generator)
    for sa_pair, cnt in instances.items():
        for group in set(splitter(sa) for sa in sa_pair):
            instance_partitioned[group][sa_pair] = cnt
    return instance_partitioned


def compare_different_alignment_solutions(first_output_path, second_output_path, external_gt=None, do_output_comparison=True):
    """
    Compare 2 different attribute clustering, returning P-R of each, and of UNION of 2.

    :param first_output_path:
    :param second_output_path:
    :param external_gt:
    :param do_output_comparison:
    :return:
    """
    # Get data of two launches AND of ground truth.
    # Also retrieve some stats on SA (most common values, frequence of attribute, cardinality...) that is in sa2other_data
    print ("Data import...")
    external_gt_path = external_gt or _config_.get_ground_truth_path()
    gt_clusters = import_cluster_detail_csv('cluster_id_real', external_gt_path)
    launch1_clusters = import_cluster_detail_csv('cluster_id', first_output_path, True)
    launch2_clusters = import_cluster_detail_csv('cluster_id', second_output_path, False)

    sa2details = launch1_clusters.sa2other_data
    # If we want to weight each event, we need to provide a weighting functions
    # Our weighting function takes minimum size of the 2 clusters
    pair_weight_function = lambda sa_pair: _get_pair_weight(sa_pair, sa2details)
    em1, em2, em_union = _compare_different_alignment_solutions_intern(launch1_clusters.sa2clusters,
                                                                       launch2_clusters.sa2clusters,
                                                                       gt_clusters.sa2clusters,
                                                                       pair_weight_function)
    if do_output_comparison:
        gt_sa2cid1 = {sa: {cid+"_sol1_compare" for cid in cids} for sa, cids in gt_clusters.sa2clusters.items()}
        gt_sa2cid2 = {sa: {cid+"_sol2_compare" for cid in cids} for sa, cids in gt_clusters.sa2clusters.items()}
        res = _build_detailed_comparison_with_ground_truth(gt_sa2cid1rr, launch1_clusters)
        res.rows.extend(_build_detailed_comparison_with_ground_truth(gt_sa2cid2, launch2_clusters).rows)
        res.export_to_csv(_config_.get_output_dir(), GROUND_TRUTH_COMPARISON_FILE_PREFIX, True)

    output = dataset.Dataset([RESULT, TP, CP, RP, P, R, F_1])
    output.add_row({RESULT: 'Solution1', TP:em1.true_positives, CP: em1.computed_positives, RP:em1.expected_positives,
                    P:em1.precision, R:em1.recall, F_1: em1.f_measure})
    output.add_row({RESULT: 'Solution2', TP:em2.true_positives, CP: em2.computed_positives, RP:em2.expected_positives,
                    P:em2.precision, R:em2.recall, F_1: em2.f_measure})
    output.add_row({RESULT: 'Union_solutions', TP:em_union.true_positives, CP: em_union.computed_positives, RP:em_union.expected_positives,
                    P:em_union.precision, R:em_union.recall, F_1: em_union.f_measure})
    output.export_to_csv(_config_.get_output_dir(), 'output_clusters_comparisons', True)

def _compare_different_alignment_solutions_intern(launch1_sa2cid:dict, launch2_sa2cid:dict, gt_sa2cid:dict, pair_weight_function=lambda sa_pair: 1):
    """
    Internal method to compare results of 2 launches
    :param launch1_sa2cid: clusters of 1st algorithm launch
    :param launch2_sa2cid: clusters of 2nd algorithm launch
    :param gt_sa2cid: ground truth cluster
    :param pair_weight_function: function that weights pair of attributes in same cluster
    :return: evaluation of each of 2
    >>> real_sa2cid = {tsa('a'): {1, 5}, tsa('b'): {1, 5}, tsa('c'): {1}, tsa('d'): {2}, tsa('e'): {2}, tsa('f'): {2},\
    tsa('g'): {1,2}, tsa('h'):{324}, tsa('kjk'):{100}, tsa('fdfsfd'):{100}}
    >>> sol1_sa2cid = {tsa('a'): {100}, tsa('b'): {100}, tsa('c'): {1100}, tsa('d'): {200}, tsa('e'): {200}, tsa('f'): {200},\
    tsa('h'): {100},  tsa('g'):{34500} , tsa('ffff'):{100}}
    >>> sol2_sa2cid = {tsa('a'): {10}, tsa('b'): {110}, tsa('c'): {10}, tsa('d'): {20}, tsa('e'): {20}, tsa('f'): {20},\
    tsa('h'): {10},  tsa('g'):{3450} , tsa('ffff'):{10}}
    >>> formula = lambda sa_pair: 10
    >>> ev1, ev2, ev_joint = _compare_different_alignment_solutions_intern(sol1_sa2cid,sol2_sa2cid,real_sa2cid,formula)
    Compute evaluation results...
    >>> (ev1.compact(), ev2.compact(), ev_joint.compact())
    ((40, 120, 60), (40, 120, 60), (50, 120, 80))
    """
    if launch1_sa2cid.keys() != launch2_sa2cid.keys():
        raise Exception("The 2 launches must provide results on same attributes")
    # Limit analysis on sa present in GT
    sas_gt = launch1_sa2cid.keys() & gt_sa2cid.keys()
    solution1_cid2sa = _invert_sa2cid(launch1_sa2cid, sas_gt)
    solution2_cid2sa = _invert_sa2cid(launch2_sa2cid, sas_gt)
    gt_cid2sa = _invert_sa2cid(gt_sa2cid, sas_gt)
    solution1_pairs = _build_all_pairs(solution1_cid2sa)
    solution2_pairs = _build_all_pairs(solution2_cid2sa)
    gt_pairs = _build_all_pairs(gt_cid2sa)
    union_solution = solution1_pairs | solution2_pairs
    print("Compute evaluation results...")
    evaluation_solution1 = build_precision_recall_from_computed_expected_positives(solution1_pairs, gt_pairs, pair_weight_function)
    evaluation_solution2 = build_precision_recall_from_computed_expected_positives(solution2_pairs, gt_pairs, pair_weight_function)
    evaluation_union = build_precision_recall_from_computed_expected_positives(union_solution, gt_pairs, pair_weight_function)
    return evaluation_solution1, evaluation_solution2, evaluation_union

# Support methods

def _build_all_pairs(cid2elements:dict, filter_element=None, pair_filter=None):
    """
    Build all attribute pairs from a dict of clusters, limiting to elements_to_evaluate if present
    :param cid2elements:
    :return:
    """
    pairs = set()
    for cluster_elements in cid2elements.values():
        if filter_element:
            cluster_elements_filtered = sorted(element for element in cluster_elements if filter_element(element))
        else:
            cluster_elements_filtered = sorted(cluster_elements)
        current_pairs = itertools.combinations(cluster_elements_filtered, 2)
        if pair_filter:
            current_pairs = [pair for pair in current_pairs if pair_filter(pair)]
        pairs.update(current_pairs)
    return pairs

def _invert_sa2cid(sa2cid, sas_list):
    """
    Invert sa2cid, limiting to provided sasa
    :param sa2cid:
    :return:
    >>> sa2cid = {tsa('a'): {1,2}, tsa('b'): {1}, tsa('c'): {3}, tsa('d'): {1}}
    >>> res = _invert_sa2cid(sa2cid, [tsa('a'), tsa('b'), tsa('c')])
    >>> sorted((a, sorted(b)) for a, b in res.items())
    [(1, [SourceAttribute(source=SourceSpecifications(site='test_site', category='dummy', pages=None), name='a'), \
SourceAttribute(source=SourceSpecifications(site='test_site', category='dummy', pages=None), name='b')]), \
(2, [SourceAttribute(source=SourceSpecifications(site='test_site', category='dummy', pages=None), name='a')]), \
(3, [SourceAttribute(source=SourceSpecifications(site='test_site', category='dummy', pages=None), name='c')])]
   """
    limited_sa2cid = {sa: cid for sa, cid in sa2cid.items() if sa in sas_list}
    return dict(bdsa_utils.build_dict(limited_sa2cid.items(), key_getter=lambda sa_cid: sa_cid[1],
                          value_getter=lambda sa_cid: sa_cid[0], multi=True,
                          key_getter_returns_multi=True))

def _sa2row(cid, real_cid, sa, other_data={}):
    row = {'source': sa.source.site, 'name': sa.name,
            'cluster_id': cid, 'cluster_id_real': real_cid}
    if sa in other_data:
        row.update(other_data[sa])
    return row

def _get_pair_weight(pair: tuple, sa2data: dict) -> int:
    """
    Get minimum nb of occurrences between a pair of  attribute (refers to original one)
    :param sa:
    :param sa2data:
    :return:
    """
    return min(int(sa2data[sa.get_original_attribute()][constants.OCCURRENCES]) for sa in pair
               if sa.get_original_attribute() in sa2data)

def build_precision_recall_from_computed_expected_positives(computed_positives, expected_positives, pair_weight_function=None,
                                                            return_true_positives=False):
    """
    Compute evaluation given computed and real positives, and a function to compute weight
    :param computed_positives:
    :param expected_positives:
    :param pair_weight_function:
    :return:
    """
    true_positives = computed_positives & expected_positives
    if pair_weight_function:
        metrics = experiment_utils.EvaluationMetrics(sum(pair_weight_function(pair) for pair in true_positives),
                                                     sum(pair_weight_function(pair) for pair in expected_positives),
                                                     sum(pair_weight_function(pair) for pair in computed_positives))
    else:
        metrics = experiment_utils.EvaluationMetrics(len(true_positives),
                                                     len(expected_positives),
                                                     len(computed_positives))
    return (metrics, true_positives) if return_true_positives else metrics

if __name__ == '__main__':
    parameter_errors = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ['part', 'comp'])
    except getopt.GetoptError:
        parameter_errors = True
    if parameter_errors or len(args) < 2:
        print("Usage: [cluster_detail_file] [ikgpp_file] (--part for partitioned evalu) "
              "(--comp for output comparison file)")
        sys.exit(2)
    partitioned = '--part' in opts
    export_comparison_file = '--comp' in opts
    res = ResultsEvaluator()
    ptt = res.launch_evaluation_files(args[0], args[1], schema_level_evaluation=SchemaLevelEvaluation.NONE,
                                do_partitioned_evaluation=partitioned,
                                output_gt_comparison=export_comparison_file)
    print(str(ptt))
