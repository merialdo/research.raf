import collections
import configparser
import glob
import itertools
import os
import re
import pandas as pd

import sys

from adapter import output_and_gt_adapter
from adapter.output_and_gt_adapter import ClusterDetailOutput, InstanceLevelClustering
from config import constants
from model import datamodel, dataset
from model.datamodel import Provenance
from scripts import results_evaluation
from scripts.results_evaluation import EvaluationResults, SchemaLevelEvaluation, ResultsEvaluator
from scripts.script_commons import get_sa2urls
from utils import io_utils, bdsa_utils, string_utils
from config.bdsa_config import _config_
from utils.bdsa_utils import getbool
from utils.experiment_utils import EvaluationMetrics

FIELDS = ['name', 'source1', 'source2', 'nb_sources', 'p', 'r', 'f1']
#'p_i', 'r_i', 'f1_i'

AgraResults = collections.namedtuple('AgraResults', FIELDS)

EvalSourceBySource = collections.namedtuple('EvalSourceBySource', 'category nb_sources nb_sources_total p r f1')


def evaluate_agra_source_by_source(agra_output_file, config_file, category):
    sa2urls = get_sa2urls(True)
    output_il = _agrawal_instance_level_output_converter(sa2urls, agra_output_file, None, category)
    output_sl = _agrawal_schema_level_output_converter(sa2urls, agra_output_file, None, category)

    sources_list_row = [row for row in io_utils.import_generic_file_per_line(config_file) if row.startswith('websitesOrdered')]
    source_list = sources_list_row[0].split('=')[1].split(',')

    old_sa2clusters = dict(output_sl.sa2clusters)


    sources_to_analyze = [source_list[0]]
    all_evals = []

    for source in source_list[1:]:
        sources_to_analyze.append(source)

        last_source = len(sources_to_analyze) == len(source_list)
        output_sl.sa2clusters = {sa: clusters for sa, clusters in old_sa2clusters.items()
                                 if sa.get_site() in sources_to_analyze}
        evaluator = results_evaluation.ResultsEvaluator(category=category)

        # Filter: there must be exactly 1 attribute from catalog out of 2 atts in the pair
        res = evaluator.launch_evaluation(output_sl, output_il, schema_level_evaluation=SchemaLevelEvaluation.NONE,
                                          do_partitioned_evaluation=False).instance_lib
        all_evals.append(EvalSourceBySource(category, len(sources_to_analyze), len(source_list),
                                            res.precision, res.recall, res.f_measure))
        if last_source:
            for name, subset in res.subsets.items():
                all_evals.append(EvalSourceBySource(category, name, len(source_list),
                                            subset.precision, subset.recall, subset.f_measure))
    io_utils.append_csv_file(['time'] + list(EvalSourceBySource._fields), [
            {
            **evaluation._asdict(),
            **{'time': string_utils.timestamp_string_format()}}
            for evaluation in all_evals],  'agrawal_partial_sources.csv')


def evaluate_agrawal(agra_output_file, do_instance_level=True, catalog_source=None, breakdown=False) -> EvaluationResults:
    """
    Evaluate agrawal results.
    :param agra_output_file: 
    :param do_instance_level: 
    :param catalog_source: if provided, we limit analysis to pairs catalog-non catalog
    :return: 
    """

    sa2urls = get_sa2urls(True)
    output_il = _agrawal_instance_level_output_converter(sa2urls, agra_output_file,
                                                         catalog_source, _config_.get_category()) \
        if do_instance_level else None
    output_sl = _agrawal_schema_level_output_converter(sa2urls, agra_output_file, catalog_source,
                                                       _config_.get_category())

    catalog_filter = None
    if catalog_source:
        catalog_filter = lambda pair: sum(1 for sa in pair if sa.source.site == catalog_source) == 1
    evaluator = results_evaluation.ResultsEvaluator(pair_filter=catalog_filter, additional_columns_schema_gt=breakdown)

    output = {}
    # Filter: there must be exactly 1 attribute from catalog out of 2 atts in the pair
    res = evaluator.launch_evaluation(output_sl, output_il, output_gt_comparison=True,
                                      do_partitioned_evaluation=False,
                                      schema_level_evaluation=SchemaLevelEvaluation.NON_WEIGHTED)
    print ('Instance-level: '+str(res.instance_lib))
    print ('Schema-level: '+str(res.schema))
    output['FULL'] = res.schema

    # Breakdown: divide GT in chunks, each with ALL external SAS and ONLY sa catalog for a particular type.
    if breakdown:
        final_data = _breakdown_evaluation(evaluator, output_sl.sa2clusters, evaluator.schema_gt.sa2clusters,
                                           evaluator.schema_gt.sa2other_data, catalog_source)
        print(bdsa_utils.dict_printer(final_data, False))
        output.update(final_data)
    io_utils.output_json_file(output, '%s_nguyen' % (_config_.get_category()), timestamp=True)
    output_pd = pd.read_json(io_utils._build_filename('%s_nguyen' % _config_.get_category(), 'json'), orient='index')
    output_pd.index.name = 'attribute_type'
    output_pd.to_csv(io_utils._build_filename('%s_nguyen' % _config_.get_category(), 'csv'), index=True)

    return res


def _breakdown_evaluation(evaluator, sa2comp_clusters, sa2exp_clusters, sa2other_gt, catalog_source):
    final_data = {}
    type2agrawal_computed_clusters = collections.defaultdict(lambda: output_and_gt_adapter.ClusterDetailOutput())
    sas_filtered_grouped = bdsa_utils.partition_data(sa2exp_clusters.keys() & sa2comp_clusters.keys(),
                                                     lambda el: el.source.site == catalog_source)
    # Start with catalog attributes
    for sa in sas_filtered_grouped[True]:
        type = sa2other_gt[sa].get('type', None)
        type2agrawal_computed_clusters[type].add_sa(sa, sa2comp_clusters[sa], sa2other_gt[sa])
    # Then add all external atts in all chunks
    for sa in sas_filtered_grouped[False]:
        for agra_output_part in type2agrawal_computed_clusters.values():
            agra_output_part.add_sa(sa, sa2comp_clusters[sa], sa2other_gt[sa])
    for type, agra_output_part in type2agrawal_computed_clusters.items():
        res = evaluator.launch_evaluation(agra_output_part, schema_level_evaluation=SchemaLevelEvaluation.NON_WEIGHTED,
                                          do_partitioned_evaluation=False)
        final_data[type] = res.schema
    return final_data


def _agrawal_schema_level_output_converter(sa2urls, filename, catalog=None, cat='camera')\
        -> ClusterDetailOutput:
    """
     Import agrawal cluster output, and convert it in form sa --> cluster_list
    :param filename:
    :return:
    """
    sa2data = {}
    sa2clusters = collections.defaultdict(set)
    for cid, sa in _import_agrawal_output(filename, cat, catalog):
        if sa.name not in _config_.get_excluded_attribute_names():
            sa2clusters[sa].add(cid)
            sa2data[sa] = {constants.OCCURRENCES: len(sa2urls[sa])}
    return ClusterDetailOutput(sa2clusters, sa2data)


def _agrawal_instance_level_output_converter(sa2urls, filename, catalog_source, category) -> InstanceLevelClustering:
    """
    Import agrawal cluster output, and convert it in form target attribute --> (name, url) pair
    :param filename:
    :return:
    """
    prov2tas = collections.defaultdict(set)
    for cid, sa in _import_agrawal_output(filename, category, catalog_source):
        for prov in (Provenance(instance.url, sa, '') for instance in sa2urls[sa]):
            prov2tas[prov].add(cid)
    return InstanceLevelClustering(prov2tas, None)


def _import_agrawal_output(filename, cat, catalog):
    """
    Generator for agrawal output
    :param filename:
    :param catalog: catalog attributes may have no source name
    :return:
    """
    output_converted = io_utils.import_json_file(filename)
    for cid, sas_string in output_converted.items():
        for sa_string in sas_string:
            # The choice of ### as separator in Agrawal is not perfect
            # as if an attribute ends with an arbitrary number of #, the # goes on
            # the beginning of source name. We have to avoid it
            sa_fixed = re.sub('(#*)###', '\\1 ###', sa_string)
            sa_fixed = sa_fixed.replace('_dot_','.')
            sa_split = sa_fixed.split('###')
            if len(sa_split) == 1:
                yield (cid, datamodel.source_attribute_factory(cat, catalog, sa_split[0].strip()))
            else:
                yield (cid, datamodel.source_attribute_factory(cat, sa_split[1].strip(), sa_split[0].strip()))

def evaluate_dir_pairs(directory, output_csv, catalog_eval=False, only_detail_file=False):
    """
    Evaluate agrawal launches on output directories, when launches are on pair of sources.
    Each directory should have the configuration file and the clusters.json output file.
    :param directory: 
    :param output_csv: 
    :param: catalog_eval: if true, evaluation is limited on pairs catalog-non catalog
    :return: 
    """
    for dir_launch in os.listdir(directory):
        output_file = os.path.join(directory, dir_launch, 'clusters.json')
        lines = io_utils.import_generic_file_per_line(glob.glob(os.path.join(directory, dir_launch, '*.properties'))[0])
        websites_row = [row.split('=')[1] for row in lines if row.split('=')[0] == 'websitesOrdered'][0]
        websites = websites_row.split(',')
        catalog_source = websites[0] if catalog_eval else None
        if only_detail_file:
            cdo = _agrawal_schema_level_output_converter(collections.defaultdict(set),
                                                         output_file, catalog_source,
                                                         _config_.get_category())
            cdo.convert_to_csv(dir_launch)
        else:
            res = evaluate_agrawal(output_file, False, catalog_source)
            agra_res = AgraResults(name=dir_launch, source1=websites[0], source2=websites[1],
                                   nb_sources= len(websites), p=res.schema.precision,
                                   r=res.schema.recall, f1=res.schema.f_measure)
            io_utils.append_csv_file(FIELDS, [agra_res._asdict()], output_csv+".csv")

if __name__ == '__main__':
    evaluate_dir_pairs(sys.argv[1], 'agrawal_evaluation',
                       getbool(sys.argv[2]), getbool(sys.argv[3]))
