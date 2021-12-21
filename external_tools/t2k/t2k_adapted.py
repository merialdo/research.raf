import collections
import csv
import getopt
import os
import shutil
import subprocess
import sys
import git
import time

import pandas
from tqdm import tqdm

from adapter.output_and_gt_adapter import ClusterDetailOutput, InstanceLevelClustering
from external_tools.t2k import json2T2K_KB, json2T2K_WB, t2k_utils
from external_tools.t2k.t2k_utils import rename_column, t2k_att2sa
from scripts import script_commons, results_evaluation
from scripts.results_evaluation import SchemaLevelEvaluation
from utils import io_utils, string_utils, bdsa_utils
from config.bdsa_config import _config_
DIR = os.path.dirname(__file__)


LITERAL_LONG = 'http://www.w3.org/2000/01/rdf-schema#Literal'

OWL_THING = 'http://www.w3.org/2002/07/owl#Thing'

LITERAL_SHORT = "rdf-schema#Literal"

NULL = 'NULL'

LABEL_KB = 'rdf-schema#label'
LABEL_KB_FULL = 'http://www.w3.org/2000/01/rdf-schema#label'
STRING_SHORT = 'XMLSchema#string'
STRING_LONG = 'http://www.w3.org/2001/XMLSchema#string'

LABEL_WT = 'label_name'
URI = 'URI'

def _reldir(adirectory):
    return os.path.join(DIR, adirectory)

OUTPUT_DIR = _reldir('temp/output')

WEBTABLE_DIR = _reldir('webtables')
KB_DIR = _reldir('dbpedia')
JAR_FILE = _reldir('t2kmatch-2.1-jar-with-dependencies.jar')

def launch_jar():
    commands = ['java', '-Xmx15G', '-cp', JAR_FILE,
                'de.uni_mannheim.informatik.dws.t2k.match.T2KMatch', '-index', _reldir('temp/index'),
    '-kb', KB_DIR, '-ontology', _reldir('OntologyDBpedia'), '-web', WEBTABLE_DIR, '-results', OUTPUT_DIR, '-verbose']
    process = subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # print(str(stdout).replace('\\t', '\t').replace('\\r\\n', '\r\n').replace('\\n', '\n'))
    print(str(stderr).replace('\\t', '\t').replace('\\r\\n', '\r\n').replace('\\n', '\n'))


def _merge_entity(entity_kb, entity_external_table):
    """
    Use kb entity, unless it is an isolated one
    """
    # If only kb row then use kb label
    if pandas.isna(entity_external_table) or entity_external_table == 'nan' or entity_external_table == '':
        return entity_kb
    else:
        if pandas.isna(entity_kb) or entity_kb.startswith('NO_'): # If only wt or both but kb is NO
            return entity_external_table
        else:
            return entity_kb



def merge_source(kb_original, wetable_datas, new_source_name, kb2webtable):

    # Remove first three lines and drop all duplicates
    kb = kb_original[3:]#
    kb.columns = list(kb_original.iloc[0])
    kb.set_index(URI, inplace=True)

    #kb = kb_original[3:].drop_duplicates(subset=[LABEL_KB_FULL]).set_index(LABEL_KB_FULL)
    # Merge columns with mappings
    joined_kb = kb.join(wetable_datas.rename(lambda x: rename_column(x, new_source_name), axis=1), how="outer")

    # Now we drop duplicates based on linkage, keeping those with least null
    joined_kb = joined_kb.iloc[joined_kb.isnull().sum(axis=1).argsort()]
    joined_kb = joined_kb[~joined_kb.index.duplicated(keep='first')]

    for kb_col, wt_col in tqdm(kb2webtable.items(), desc='Merging sources....'):
        joined_kb[kb_col] = joined_kb.apply(
            lambda row: _convert_column_value(row[kb_col], row[wt_col]), axis=1)
        joined_kb.drop(wt_col, inplace=True, axis=1)

    # Merge Entity ID
    label_ext_column = rename_column(LABEL_WT, new_source_name)
    joined_kb[LABEL_KB_FULL] = joined_kb.apply(lambda row: _merge_entity(row[LABEL_KB_FULL], row[label_ext_column]), axis=1)

    ## Now again in T2K format: fix columns
    joined_kb.reset_index(inplace=True)
    joined_kb.rename(columns={'index': URI}, inplace=True)
    cols = joined_kb.columns.to_list()
    cols = [URI, LABEL_KB_FULL] + cols[2:]
    joined_kb = joined_kb[cols]

    #Add again top rows
    row_fullname = {x: x for x in cols}
    row_type_short = {**{URI: URI, LABEL_KB_FULL: LITERAL_SHORT}, **{x: STRING_SHORT for x in cols[2:]}}
    row_type_long = {**{URI: OWL_THING, LABEL_KB_FULL: LITERAL_LONG}, **{x: STRING_LONG for x in cols[2:]}}
    final_kb = pandas.concat([pandas.DataFrame([row_fullname, row_type_short, row_type_long]), joined_kb],
                             ignore_index=True)
    new_column_headers = [URI, LABEL_KB] + [x.split('/')[-1] for x in cols[2:]]
    final_kb.to_csv(os.path.join(KB_DIR, 'dbpedia.csv'), header=new_column_headers, index=False, sep=',', na_rep="",
                    quotechar='"', quoting=csv.QUOTE_ALL,  escapechar='\\')

    print('KB merged and output.')
    return final_kb


def _convert_column_value(old_col_value, new_col_value):
    if pandas.isna(old_col_value) or old_col_value == 'nan' or old_col_value == '':
        if pandas.isna(new_col_value) or new_col_value == 'nan' or new_col_value == '':
            return ''
        else:
            return new_col_value

    else:
        return old_col_value


def get_schema_mappings_output(output_dir, wt, source, score_threshold=0.2):
    """
    Get the schema mappings given by t2k
    """
    mappings_csv = pandas.read_csv(os.path.join(output_dir, 'schema_correspondences.csv'),
                                   names=['wt_num', 'kb', 'score'])

    filtered_mappings = mappings_csv[mappings_csv['score'] >= score_threshold]
    # -1 because label is no longer a column
    filtered_mappings['wt'] = filtered_mappings['wt_num'].apply(
        lambda x: t2k_utils.rename_column(wt.columns[_extract_column_id(x)], source))
    # Remove mappings referring to uri or label
    uri_col = t2k_utils.rename_column(URI, source)
    label_name_col = t2k_utils.rename_column(LABEL_WT, source)
    filtered_mappings = filtered_mappings[~filtered_mappings.wt.isin([uri_col, label_name_col]) &
                                          ~filtered_mappings.kb.isin([URI, LABEL_KB_FULL])]
    return dict(zip(filtered_mappings['kb'], filtered_mappings['wt'])) #, dict(zip(filtered_mappings['wt'], filtered_mappings['kb']))


def _rename_linked_rows(output_dir, wt, source, score_threshold=0.5):
    """
    Rows in wt linked to rows in KB according to T2K will get the KB URLs
    """
    mappings_csv = pandas.read_csv(os.path.join(output_dir, 'instance_correspondences.csv'),
                                   names=['wt_row_num', 'kb_uri', 'score'])

    filtered_mappings = mappings_csv[mappings_csv['score'] >= score_threshold]
    # -1 because label is no longer a column
    filtered_mappings['wt_uri'] = filtered_mappings['wt_row_num'].apply(
        lambda x: wt.iloc[_extract_row_id(x)][URI])
    wt_uri2kb_uri = dict(zip(filtered_mappings['wt_uri'], filtered_mappings['kb_uri'])) #, dict(zip(filtered_mappings['wt'], filtered_mappings['kb'])
    wt[URI] = wt[URI].apply(lambda wturi: wt_uri2kb_uri.get(wturi, wturi))
    wt.set_index(URI, inplace=True)


def _extract_column_id(col_label):

    """
    The schema correspondences column label is Col~5, so we extract 5
    """
    return int(col_label.split('Col')[1])

def _extract_row_id(col_label):

    """
    The schema correspondences column label is Row~5, so we extract 5
    """
    return int(col_label.split('Row')[1])



def t2k_adapted(dataset_name, category, ground_truth_name, complete=True):
    """
    T2K launched iteratively to merge each source one by one
    """
    dataset_directory = _config_.get_spec_path_from_dataset_name(dataset_name)
    sources_order = _get_source_list(category, dataset_directory)
    first_source_dir = os.path.join(dataset_directory, sources_order[0])
    sources_managed = [sources_order[0]]  # List of sources that have been integrated

    # Produce KB table
    kb_current = json2T2K_KB.produce_output(first_source_dir, sources_order[0], category, KB_DIR)
    target_att2source_atts = collections.defaultdict(set)

    seconds = 0
    # Cycle on all other sources
    for source in sources_order[1:]:
        next_source_dir = os.path.join(dataset_directory, source)
        if source != '\n' and os.path.exists(os.path.join(next_source_dir, '%s_spec.json' % category)):
            # Generate web table input and launch T2K
            wt = json2T2K_WB.produceOutput(next_source_dir, category, WEBTABLE_DIR, source)
            print('T2K on source %s' % source)
            _remove_old_cache_data()
            now = time.time()
            launch_jar()
            end = time.time()
            seconds += (end - now)
            print('T2K ended. Cumulated time %d ' % seconds)

            target_att2current_source_atts = get_schema_mappings_output(OUTPUT_DIR, wt, source)
            _rename_linked_rows(OUTPUT_DIR, wt, source)
            _build_chained_match(target_att2current_source_atts, target_att2source_atts)
            # This will merge kb with webtables directly from CSV output
            kb_current = merge_source(kb_current, wt, source, target_att2current_source_atts) #get source2
            sources_managed.append(source)

    _output_t2k_results_evaluation(dataset_name, target_att2source_atts, category, sources_managed, seconds,
                                   ground_truth_name, complete)


def _get_source_list(category, sources_directory):
    source_by_link_location = os.path.join(sources_directory, 'sources_by_linkage_%s.txt' % category)
    if not (os.path.exists(source_by_link_location)):
        source_by_link_location = os.path.join(sources_directory, 'sources_by_linkage.txt')
    # Get ordered sources and the first one to start
    sources_order = [source.split('__')[0] for source in io_utils.import_generic_file_per_line(source_by_link_location)]
    return sources_order


def _output_t2k_results_evaluation(dataset_name: str, target_att2source_atts: dict, category: str, sources_managed:list,
                                   seconds, ground_truth_name=None, complete=True):
    """
    Output results: in a TXT file, serialized data, evaluate results that will be print and added to a CSV file
    """

    hashed_list_sources = string_utils.compute_string_hash(str(sources_managed))[:8]
    git_id, git_message = bdsa_utils.get_git_commit()

    sa2urls = script_commons.get_sa2urls(True, dataset_name=dataset_name, category=category)
    nb_sources_category = len({sa.source for sa in sa2urls.keys()})
    number_sources = str(len(sources_managed)) if len(sources_managed) < nb_sources_category else 'ALL'

    label_launch = 't2k_%s_%s_%s-sources_%s-git_%s-sources' % (dataset_name, category, number_sources,
                                                               git_id, hashed_list_sources)
    io_utils.output_txt_file({target: ', '.join(ls)
                              for target, ls in target_att2source_atts.items()},
                             label_launch, timestamp=False)
    io_utils.output_ser_file(target_att2source_atts, label_launch + '.ser', False)

    ground_truth_file = ground_truth_name or dataset_name

    if complete:
        final_res = _t2k_evaluation(category,  ground_truth_file, set(sources_managed), target_att2source_atts, sa2urls,
                                    False)
        print(final_res)
        io_utils.append_csv_file(['time', 'dataset', 'category', 'P', 'R', 'F1', 'nb_sources', 'nb_sources_cat',
                                  'git_message',  'sources_hex', 'git_hex', 'seconds'],
                                 [{'time': string_utils.timestamp_string_format(), 'dataset': dataset_name,
                                   'category': category, 'P': final_res.precision, 'R': final_res.recall,
                                   'F1': final_res.f_measure, 'nb_sources': number_sources,
                                   'nb_sources_cat': nb_sources_category, 'sources_hex': hashed_list_sources + '-H',
                                   'git_hex': git_id + '-H', 'git_message': git_message, 'seconds': seconds}], 't2k_results.csv')
    else:
        all_evals = evaluate_source_by_source(category, ground_truth_file, sources_managed, target_att2source_atts, sa2urls)
        io_utils.append_csv_file(['time'] + list(EvalSourceBySource._fields) + ['git_message',  'sources_hex', 'git_hex'], [
            {
            **evaluation._asdict(),
            **{'time': string_utils.timestamp_string_format(), 'sources_hex': hashed_list_sources + '-H',
               'git_hex': git_id + '-H', 'git_message': git_message}}
            for evaluation in all_evals],  'partial_sources.csv')


def _t2k_evaluation(category, ground_truth_file, sources_managed:set,
                    target_att2source_atts, sa2urls, partitioned_evaluation=False):
    # Convert results to instance-level
    schema_level = ClusterDetailOutput()
    instance_level = InstanceLevelClustering()
    sas_not_matched = set(sa for sa in sa2urls.keys() if
                          sa.get_site() in sources_managed)  # We have to add all SAs that we consider in schema level.
    for ta, source_atts in target_att2source_atts.items():
        ta_name = str(ta)
        for t2k_sa in source_atts:
            sa = t2k_att2sa(t2k_sa, category)
            sas_not_matched.discard(sa)
            schema_level.add_sa(sa, ta_name)
            instance_level.add_sa_full(ta_name, sa, sa2urls)
        original_sa = t2k_att2sa(ta, category)
        sas_not_matched.discard(original_sa)
        schema_level.add_sa(original_sa, ta_name)
        instance_level.add_sa_full(ta_name, original_sa, sa2urls)
    for sa in sas_not_matched:
        schema_level.add_sa(sa, str(sa))
    # Evaluate
    evaluation = results_evaluation.ResultsEvaluator(
        _config_.get_ground_truth_path_from_dataset_name(ground_truth_file),
        _config_.get_ground_truth_instance_level_path_from_dataset_name(ground_truth_file), category)
    res = evaluation.launch_evaluation(schema_level, instance_level, SchemaLevelEvaluation.NONE,
                                       do_partitioned_evaluation=partitioned_evaluation)

    final_res = res.instance_lib
    return final_res


def evaluate_cached_data(cached_filename:str, dataset_name, category, ground_truth_name=None, complete=False):
    """
    Evaluate a result that has been already cached
    """
    target2sas = io_utils.import_ser_file(cached_filename)

    dataset_directory = _config_.get_spec_path_from_dataset_name(dataset_name)
    sources_order = _get_source_list(category, dataset_directory)

    _output_t2k_results_evaluation(dataset_name, target2sas, category, sources_order, ground_truth_name, complete)


EvalSourceBySource = collections.namedtuple('EvalSourceBySource', 'category dataset nb_sources nb_sources_total p r f1')


def evaluate_source_by_source(category, ground_truth_file, sources_managed:list, target_att2source_atts, sa2urls):
    sources_to_evaluate = {sources_managed[0]}
    # Not optimized but quicker to code :)
    all_evals = []
    for source in sources_managed[1:]:
        sources_to_evaluate.add(source)
        last_source = len(sources_to_evaluate) == len(sources_managed)
        partial_t2s = {ta: {sa for sa in sas if t2k_att2sa(sa, category).get_site() in sources_to_evaluate}
                       for ta, sas in target_att2source_atts.items()}
        res = _t2k_evaluation(category, ground_truth_file, sources_to_evaluate, partial_t2s, sa2urls,
                              partitioned_evaluation=last_source)
        all_evals.append(EvalSourceBySource(category, ground_truth_file, len(sources_to_evaluate), len(sources_managed),
                                            res.precision, res.recall, res.f_measure))
        if last_source:
            for name, subset in res.subsets.items():
                all_evals.append(EvalSourceBySource(category, ground_truth_file, name, len(sources_managed),
                                            subset.precision, subset.recall, subset.f_measure))
    return all_evals

def _build_chained_match(target_att2current_source_atts, target_att2source_atts):
    """
    Merge old matches with new, to build a complete chain of matches
    """
    for target, source_att in target_att2current_source_atts.items():
        target_att2source_atts[target].add(source_att)

def _remove_old_cache_data():
    if os.path.exists(_reldir('temp')):
        shutil.rmtree(_reldir("temp"))
    if os.path.exists(_reldir('dbpedia.bin')):
        os.remove(_reldir('dbpedia.bin'))

enumerate

if __name__ == '__main__':
    # https://www.tutorialspoint.com/python/python_command_line_arguments.htm
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ['cached=', 'partial'])
    except getopt.GetoptError:
        print('Wrong format')
        sys.exit(2)

    dataset_name = args[0]
    category = args[1]
    if len(args) > 2:
        ground_truth_name = args[2]
    else:
        ground_truth_name = dataset_name

    cached_name = None
    complete = True

    for opt, arg in opts:
        if opt == '--cached':
            cached_name = arg
        if opt == '--partial':
            complete = False

    if cached_name:
        evaluate_cached_data(cached_name, dataset_name, category, ground_truth_name, complete)
    else:
        t2k_adapted(dataset_name, category, ground_truth_name, complete)
     #launch_jar()
