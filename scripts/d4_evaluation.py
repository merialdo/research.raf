import collections
import os

import pandas

from adapter import output_and_gt_adapter
from adapter.output_and_gt_adapter import InstanceLevelClustering, ClusterDetailOutput
from model import datamodel, dataset
from model.datamodel import Provenance
from scripts import script_commons, results_evaluation
from scripts.results_evaluation import SchemaLevelEvaluation, ResultsEvaluator
from utils import io_utils, experiment_utils
from config.bdsa_config import _config_
from enum import Enum

SPACE_NAME = '_SPACE_'

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Method(Enum):
    RAF = 1
    D4 = 2


def evaluate_method(output_dir, method: Method, category=None, dataset_dir=None, il_gt=None, sl_gt=None, part_eval=False,
                          standard_raf_eval=True, target_specific_d4_eval=False):
    """
    Evalute D4 or RAF
    """
    if method == Method.D4:
        il_output, sl_output = d4_instance_level_converter(output_dir, category)
    else:
        il_output, sl_output = _raf_instance_level_converter(output_dir, category)
    evaluator = adapt_config_get_evaluator(category, dataset_dir, il_gt, sl_gt)
    raf_eval = None
    ta2eval = None
    if standard_raf_eval:
        raf_eval = evaluator.launch_evaluation(sl_output, il_output, schema_level_evaluation=SchemaLevelEvaluation.NONE,
                                      do_partitioned_evaluation=part_eval)
    if target_specific_d4_eval:
        ta2eval = eval_target_per_target(il_output, evaluator)
    return raf_eval, ta2eval


def d4_evaluator_full(evaluate_classic=True, evaluate_target_atts=False):
    """
    Evaluate different results of D4 and put it in a CSV
    :param evaluate_classic:

    :param evaluate_target_atts:
    :return:
    """
    info = dataset.import_csv(os.path.join(THIS_DIR, 'd4_eval_list.csv'))
    if evaluate_classic:
        output_eval = dataset.Dataset(['name', 'category', 'P', 'R', 'F1'])
    if evaluate_target_atts:
        output_target_atts_eval = dataset.Dataset(['name', 'category', 'ta', 'p', 'r', 'f1'])

    for row in info.rows:
        category_input = row['category']
        raf_eval, d4_eval = evaluate_method(
            row['directory'], Method[row['type'].upper()], category_input, row['datadir'],
            row['instance_level'], row['schema_level'], False, evaluate_classic, evaluate_target_atts)
        if evaluate_classic:
            output_eval.add_row({'name': row['name'], 'category': category_input,
                            'P': raf_eval.instance_lib.precision, 'R': raf_eval.instance_lib.recall,
                            'F1': raf_eval.instance_lib.f_measure})
        if evaluate_target_atts:
            for ta, eval in d4_eval.items():
                output_target_atts_eval.add_row({'name': row['name'], 'category': category_input,
                                                 'ta': str(ta), 'p': eval.precision, 'r': eval.recall,
                                                 'f1': eval.f_measure})

    if evaluate_classic:
        output_eval.export_to_csv(_config_.get_output_dir(), 'd4_evaluation.csv', True)

    if evaluate_target_atts:
        output_target_atts_eval.export_to_csv(_config_.get_output_dir(), 'eval_by_ta', True)


def _raf_instance_level_converter(algo_output_dir, category_input=None):
    category = category_input or _config_.get_category()
    d4_instance_level = output_and_gt_adapter.build_il_clustering_from_ikgpp(
        algo_output_dir + '/results_instances.json', category)
    d4_schema = output_and_gt_adapter.import_cluster_detail_csv('cluster_id',
                                                                algo_output_dir + '/results_schema.json',
                                                                category_input=category)
    return d4_instance_level, d4_schema


def adapt_config_get_evaluator(category_input=None, datadir=None, instance_level_filename=None,
                               schema_level_filename=None):
    """
    Get evaluator on given data, potentially changing config
    :param category_input:
    :param datadir:
    :param instance_level_filename:
    :param schema_level_filename:
    :return:
    """
    schema_level = _config_.get_file_relative_to_input('ground_truth/' + schema_level_filename) if \
        schema_level_filename else None
    instance_level = _config_.get_file_relative_to_input('ground_truth/' + instance_level_filename) if \
        instance_level_filename else None
    dataset_dir = _config_.get_file_relative_to_input('dataset/' + datadir) if datadir else None
    if dataset_dir:
        _config_.config.set('inputs', 'specifications_full_path', dataset_dir)
    if category_input:
        _config_.config.set('inputs', 'category', category_input)
    eval = results_evaluation.ResultsEvaluator(category=category_input, schema_level_gt_path=schema_level,
                                               instance_level_gt_path=instance_level)
    return eval


def d4_instance_level_converter(directory, category=_config_.get_category()):
    """
    Import d4 output, and convert it in form InstanceLevelClustering
    :param filename:
    :return:
    """

    # Our final output is correspondences between att occs (==provenance) and cluster id.
    prov2cluster_id = collections.defaultdict(set)
    # We need list of specs in which each att occurs to convert D4 schema-level results to instance-level
    sa2urlvalue = script_commons.get_sa2urls(True, provide_value=True)

    # The schema-level output is also needed for legacy issues
    sa2clusters = {sa: set() for sa in sa2urlvalue.keys()}

    domains = os.path.join(directory, 'domains')
    # Browse output, each json file is a domain i.e. a cluster of atts
    for json_domain_file in io_utils.browse_directory_files(domains):
        json_domain = io_utils.import_json_file(json_domain_file.path)
        cid = json_domain_file.name.replace('.json', '')

        ## This are all the terms extracted
        valid_terms = {term['name'] for termlist in json_domain['terms'] for term in termlist}

        for d4_att in json_domain['columns']:
            sa = datamodel.source_attribute_factory(category, d4_att['dataset'],
                                                    d4_att['name'].replace(SPACE_NAME, ' ').lower())
            if sa in sa2urlvalue:
                sa2clusters[sa].add(cid)
                for url_value in sa2urlvalue[sa]:
                    if url_value.value.upper() in valid_terms:
                        prov2cluster_id[Provenance(url_value.url, sa, None)].add(cid)
    return InstanceLevelClustering(prov2cluster_id, None), ClusterDetailOutput(sa2clusters,
                                                                               collections.defaultdict(dict))


def d4_instance_level_converter_old027(directory, category=_config_.get_category()):
    """
    Import d4 output, and convert it in form InstanceLevelClustering
    :param filename:
    :return:
    """

    # Our final output is correspondences between att occs (==provenance) and cluster id.
    prov2cluster_id = collections.defaultdict(set)
    # We need list of specs in which each att occurs to convert D4 schema-level results to instance-level
    sa2urls = script_commons.get_sa2urls(True)

    # The schema-level output is also needed for legacy issues
    sa2clusters = {sa: set() for sa in sa2urls.keys()}

    # Browse output, each json file is a domain i.e. a cluster of atts
    for json_domain_file in io_utils.browse_directory_files(directory):
        json_domain = io_utils.import_json_file(json_domain_file.path)
        cid = json_domain_file.name.replace('.json', '')
        for d4_att in json_domain['columns']:
            sa = datamodel.source_attribute_factory(category, d4_att['dataset'],
                                                    d4_att['name'].replace(SPACE_NAME, ' ').lower())
            _convert_single_schema_level_matching_to_instance(cid, prov2cluster_id, sa, sa2urls)
            if sa in sa2urls:
                sa2clusters[sa].add(cid)
    return InstanceLevelClustering(prov2cluster_id, None), ClusterDetailOutput(sa2clusters,
                                                                               collections.defaultdict(dict))


def _convert_single_schema_level_matching_to_instance(cid, prov2cluster_id, sa, sa2urls):
    """
    Add a single sa 2 cluster_id match to N instance-level matches (one per url <-> specification)
    :param cid:
    :param prov2cluster_id:
    :param sa:
    :param sa2urls:
    :return:
    """
    if sa in sa2urls:
        for url_value in sa2urls[sa]:
            prov2cluster_id[Provenance(url_value.url, sa, None)].add(cid)


def eval_target_per_target(instance_result: InstanceLevelClustering, evaluator: ResultsEvaluator):
    exp_ta2prov = collections.defaultdict(set)
    comp_ta2prov = collections.defaultdict(set)
    exp_ta2comp_ta_overlapping = collections.defaultdict(set)

    # Find all overlapping TAS
    for prov, gt_tas in evaluator.instance_gt.prov2ta.items():
        for gt_ta in gt_tas:
            exp_ta2prov[gt_ta].add(prov)
        for computed_ta in instance_result.prov2ta[prov]:
            comp_ta2prov[computed_ta].add(prov)
            for gt_ta in gt_tas:
                exp_ta2comp_ta_overlapping[gt_ta].add(computed_ta)

    # Fina data
    ta2eval = {}
    for exp_ta, comp_tas in exp_ta2comp_ta_overlapping.items():
        comp_tas2eval = {}
        provs_expected = exp_ta2prov[exp_ta]
        len_provs_expected = len(provs_expected)
        for comp_ta in comp_tas:
            provs_comp = comp_ta2prov[comp_ta]
            common = provs_comp & provs_expected
            eval = experiment_utils.EvaluationMetrics(len(common), len_provs_expected, len(provs_comp))
            comp_tas2eval[comp_ta] = eval
        best_eval = max(comp_tas2eval.items(), key=lambda ta2eval: ta2eval[1].f_measure)
        ta2eval[exp_ta] = best_eval[1]

    return ta2eval


if __name__ == '__main__':
    d4_evaluator_full()
