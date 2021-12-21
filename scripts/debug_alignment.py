#coding: utf-8
import itertools
import math

import networkx as nx

from adapter import global_file_linkage, adapter_factory
from config.bdsa_config import _config_
from model import datamodel
from utils import graph_utils

from utils.experiment_utils import EvaluationMetrics
from pipeline.pipeline_analyzer import *

CLASSIFICATION = 'classification'

FN = 'FN'
FP = 'FP'
TP = 'TP'
F2_MEASURE = 'F2-Measure'
F_MEASURE = 'F-Measure'
RECALL = 'Recall'
PRECISION = 'Precision'
VALID = '% valid'
NUMBER_ELEMENTS = 'Number elements'

TODO = 'TODO'
NO_AUTO = 'NO_AUTO'

AUTOMATIC_TAG = 'automatic_tag'
MANUAL_TAG = 'manual_tag'
ATTRIBUTE_2 = 'attribute_2'
SITE_2 = 'site_2'
ATTRIBUTE_1 = 'attribute_1'
SITE_1 = 'site_1'
CATEGORY = 'category'
MAX_SIZE = 'max_size'
MIN_SIZE = 'min_size'
ID = 'id'

def build_simplified_cluster_detail_output(output_file):
    """
    The cluster detail output may be difficult to read if many clusters are overlapped (eg ABC, ABD, BCD...) and/or if
    there are a lot of attributes with same prefix in same cluster. This method simplifies the output by keeping max
    1 attribute per prefix, and by giving higher numbers to smaller clusters that contains attributes already clustered.
    :return:
    """
    ds = dataset.import_csv(output_file)
    cid2sas = collections.defaultdict(set)
    sa2row = {}
    others_row = []
    all_sas = set()
    for row in ds.rows:
        cid = row['cluster_id']
        sa = datamodel.source_attribute_factory('dummy', row['source'], row['full_name'])
        if not sa.is_generated():
            sa2row[sa] = row
        if cid != '' and not cid.startswith('ISOLATED'):
            cid2sas[cid].add(sa.get_original_attribute())
            all_sas.add(sa.get_original_attribute())
        else:
            others_row.append(row)
    already_computed = set()
    new_cid_counter = 0
    ds_output = dataset.Dataset(ds.ud_headers)
    while len(cid2sas) > 0:
        cid, sas = max(cid2sas.items(), key=lambda cluster: len(cluster[1] - already_computed))
        for sa in sas:
            row = dict(sa2row[sa])
            row['cluster_id'] = new_cid_counter
            ds_output.add_row(row)
        already_computed.update(sas)
        #If we already put all attributes, then we reset and start again
        if already_computed == all_sas:
            already_computed.clear()
        del cid2sas[cid]
        new_cid_counter += 1
    for row in others_row:
        ds_output.add_row(row)
    ds_output.export_to_csv(_config_.get_output_dir(), 'simplified_output', True)

def build_debug_table(category, source2attributes, do_csv_vs_json=True, show_null=True):
    """
    Compare values of attributes from different sources, grouping in a single row source pages in linkage
    :param category:
    :param source2attributes: 
    :return: 
    """
    #Loading all sources
    pid2sa2values = collections.defaultdict(dict)
    isolated_index = 0
    source_linkage = adapter_factory.linkage_factory(_config_.get_linkage_suffix())

    for source_label in list(source2attributes.keys()):
        source = adapter_factory.spec_factory().source_specifications(source_label, category)
        attributes = set(source2attributes[source_label])
        if len(attributes) == 0:
            attributes = set().union(key for key2values in source.pages.values() for key in key2values.keys())
        for url, specifications in source.pages.items():
            ids = source_linkage.ids_by_url(url, source.site, category)
            if len(ids) == 0:
                ids = ['ISOLATED_%d' % isolated_index]
                isolated_index += 1

            for key in attributes:
                # Value must be provided only if it exists an attribute with that name (no NULL), or if show null is activated
                if show_null or key in specifications:
                    value = specifications[key] if key in specifications else 'NULL'
                    for pid in ids:
                        pid2sa2values[pid][datamodel.SourceAttribute(source.metadata_only(), key)] = value

    # build table using linkage
    if do_csv_vs_json:
        table = dataset.Dataset([ID])
        for pid, sa2values in pid2sa2values.items():
            #build single row
            row = dict(sa2values)
            row[ID] = pid
            table.add_row(row)

        filename = 'compare_%s_%s'%(category, '__'.join(source2attributes.keys())[:30])
        table.export_to_csv(_config_.get_output_dir(), filename, True)
    else:
        pid2sa2val_serializable = {pid: {str(sa): value for sa, value in sa2value.items()} for pid, sa2value in pid2sa2values.items()}
        io_utils.output_json_file(pid2sa2val_serializable, 'debug_table')



def attributes_analysis(category, source2attributes: dict):
    """
    Analyze attribute values
    :param category:
    :param source2attributes:
    :return:
    """
    values = set(['NULL'])
    attributes2counter = collections.defaultdict(collections.Counter)
    for source_label, atts in source2attributes.items():
        source = adapter_factory.spec_factory().source_specifications(source_label, category)
        number_pages = len(source.pages)
        do_all_atts = len(atts) == 0
        for key2values in source.pages.values():
            for att, val in key2values.items():
                if do_all_atts or att in atts:
                    attributes2counter[datamodel.source_attribute_factory(category, source_label, att)][val] += 1
                    values.add(val)
        for att, cnt in attributes2counter.items():
            null_size = number_pages - sum(cnt.values())
            if null_size > 0:
                cnt['NULL'] = null_size


    output = dataset.Dataset()
    for value in sorted(values):
        row = {}
        for att, counter in attributes2counter.items():
            if value in counter:
                row[str(att)] = '%s_%s' % (value, counter[value])
        output.add_row(row)
    output.export_to_csv(_config_.get_output_dir(), 'attributes_analysis', True)

def build_golden_set(number_of_source_pairs=40, do_build_debug_table=False):
    """
    Build a sample set of sources and attributes, useful for evaluation. 
    :param number_of_source_pairs: 
    :param do_build_debug_table: 
    :return: 
    """
    spec_adapter=adapter_factory.spec_factory()
    linkage = global_file_linkage.GlobalLinkageFile()
    g = _build_common_ids_graph(spec_adapter, linkage, True)
    print ("**** SELECT Sample sources ****")
    selected_source_pairs = stats_utils.sample_dataset(3, number_of_source_pairs, g.edges(data=True),
                                                      [
                                   stats_utils.NamedLambda(lambda _edge: _safe_log(_edge[2][MIN_SIZE]), "Size of smaller source"),
                                   stats_utils.NamedLambda(lambda _edge: _safe_log(_edge[2][MAX_SIZE]), "Size of bigger source"),
                                   stats_utils.NamedLambda(lambda _edge: _safe_log(_edge[2][constants.WEIGHT]), "Common IDS")
                                ],
                                [
                                    stats_utils.NamedLambda(lambda _edge: _edge[2][CATEGORY], "Category")
                                ], True
                                                       )

    #selected_sources = set(source for pair in ([pair[0], pair[1]] for pair in selected_source_pairs) for source in pair)
    sample_set = dataset.Dataset([CATEGORY, SITE_1, SITE_2, ATTRIBUTE_1, ATTRIBUTE_2, MANUAL_TAG, AUTOMATIC_TAG])
    for source_pair in tqdm(selected_source_pairs, desc='Build attribute pairs table...'):
        _add_gs_rows(spec_adapter, do_build_debug_table, sample_set, source_pair)

    sample_set.export_to_csv(_config_.get_output_dir(), 'yes_or_no', True)


def _add_gs_rows(spec_adapter, do_build_debug_table, sample_set, source_pair):
    """
    Add golden set rows for a pair of sources.
    - Computes cartesian product of attributes
    - Filter pairs that are clearly different
    - All other pairs are added as rows to dataset
    
    :param do_build_debug_table: 
    :param sample_set: the dataset to which add elements
    :param source_pair: couple of sources
    :return: 
    """
    a2c_s1 = _compute_attribute2counter(source_pair[0])
    a2c_s2 = _compute_attribute2counter(source_pair[1])
    att_pairs = itertools.product(a2c_s1.keys(), a2c_s2.keys())
    if do_build_debug_table:
        build_debug_table(source_pair[0].category, {
            source_pair[0].site: a2c_s1.keys(),
            source_pair[1].site: a2c_s2.keys()})
    nb_att_pairs = len(a2c_s1.keys()) * len(a2c_s2.keys())
    for att_pair in att_pairs:
        value_set1 = a2c_s1[att_pair[0]]
        value_set2 = a2c_s2[att_pair[1]]
        jaccard_similarity = float(sum((value_set1 & value_set2).values())) / sum((value_set1 | value_set2).values())
        if jaccard_similarity >= 0.2:
            sample_set.add_row({
                CATEGORY: source_pair[0].category, SITE_1: source_pair[0].site, ATTRIBUTE_1: att_pair[0],
                SITE_2: source_pair[1].site, ATTRIBUTE_2: att_pair[1], MANUAL_TAG: TODO, AUTOMATIC_TAG: TODO
            })


def _compute_attribute2counter(sp0, spec_adapter):
    """
    Builds a list with attributes 2 multiset of its values
    :param sp0: 
    :return: 
    """
    source_att1 = spec_adapter.source_specifications(sp0.site, sp0.category, normalize_key=True,
                                                normalize_value=True)
    attribute2counter = collections.defaultdict(lambda: collections.Counter())
    for specs in source_att1.pages.values():
        for att, value in specs.items():
            attribute2counter[att].update([value])
    return attribute2counter


def _build_common_ids_graph(spec_adapter, linkage : global_file_linkage.GlobalLinkageFile, remove_edge_if_no_linkages=False):
    url2sources = {}
    #label2sources = {}
    g = nx.Graph()
    for source in spec_adapter.specifications_generator(False, False):
        source_md = source.metadata_only()
        g.add_node(source_md, **{constants.WEIGHT: len(source.pages)})
        for url in source.pages:
            url2sources[url] = source_md

    #TODO find a more efficient way
    for x,y in tqdm(itertools.combinations(g.nodes(data=True), 2), desc='Initialize nodes'):
        if x[0].category == y[0].category:
            max_source = max(x[1][constants.WEIGHT], y[1][constants.WEIGHT])
            min_source = min(x[1][constants.WEIGHT], y[1][constants.WEIGHT])
            g.add_edge(x[0], y[0], **{constants.WEIGHT: 0, CATEGORY: x[0].category,
                                      MAX_SIZE: max_source, MIN_SIZE: min_source})

    for cat2urls in list(linkage.get_full_map().values()):
        for urls in list(cat2urls.values()):
            for url1, url2 in itertools.combinations(urls, 2):
                if url1 in url2sources and url2 in url2sources:
                    source1 = url2sources[url1]
                    source2 = url2sources[url2]
                    if source1 != source2 and source1.category == source2.category:
                        graph_utils.increment_edge_attribute(g, source1, source2, constants.WEIGHT, 1)

    if remove_edge_if_no_linkages:
        for edge in list(g.edges(data=True)):
            if edge[2][constants.WEIGHT] == 0:
                g.remove_edge(edge[0], edge[1])

    return g

def compare_with_onto(alignment_file_path, ontology_file_path):
    """
    Cf. _compare_with_ontology_intern
    :param alignment_file:
    :param ontology_file:
    :return:
    """

    alignment_data = dataset.import_csv(alignment_file_path)
    stats_utils.compute_head_tail_dataset(alignment_data, OCCURRENCES)

    ontology_data = dataset.import_csv(ontology_file_path)
    ontology = [element[constants.NAME] for element in ontology_data.rows]
    comparison, metrics = _compare_with_ontology_intern(alignment_data.rows, ontology)
    alignment_data.add_element_to_header(CLASSIFICATION)
    alignment_data.add_element_to_header(constants.NAME)
    alignment_data.export_to_csv(_config_.get_output_dir(), 'ontology_comparison', True)
    print("Metrics: %s"%(str(metrics)))

def _compare_with_ontology_intern(algo_output:list, ontology:list) -> (list, EvaluationMetrics) :
    """
    Compares output of algorithm with an external ontology.
    Matches an element if it corresponds with one out of 3 most frequent attribute name in cluster.
    Produces a CSV in output with all matches and all mismatches.
    :return:

    >>> algo_output = [{constants.CLUSTER_ID:1, TOP_1:'opt zoom', TOP_2:'zoom', TOP_3:'optical zoom', OCCURRENCES:100},\
    {constants.CLUSTER_ID:2, TOP_1:'width', TOP_2:'larghezza', TOP_3:'optical zoom', OCCURRENCES:50},\
    {constants.CLUSTER_ID:3, TOP_1:'were this helpful?', TOP_2:'where', TOP_3:'this', OCCURRENCES:20},\
    {constants.CLUSTER_ID:4, TOP_1:'width', TOP_2:'where', TOP_3:'fdfsdf', OCCURRENCES:5}]
    >>> ontology = ['optical zoom','brand', 'width', 'weight']
    >>> output, metrics = _compare_with_ontology_intern(algo_output, ontology)
    >>> (metrics.precision, metrics.recall)
    (0.6666666666666666, 0.5)
    >>> output[4]
    {'cluster_id': 'NA', 'name': 'brand', 'classification': 'FN'}
    """
    name2cids = [collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)]

    computed_positives = len(algo_output)
    expected_positives = len(ontology)
    tp = 0

    # Do this in 3 different iterations, in order to make sure top1 are prefered over top2 and 3 (is there a better way?)
    for element in algo_output:
        element[CLASSIFICATION] = 'FP'
        name2cids[0][string_utils.normalize_keyvalues(element[TOP_1])].append(element)
        name2cids[1][string_utils.normalize_keyvalues(element[TOP_2])].append(element)
        name2cids[2][string_utils.normalize_keyvalues(element[TOP_3])].append(element)
    # For each ontology term we look for it in cluster names, starting from TOP1.
    for i in range(3):
        for onto_element in list(ontology):
            onto_clean = string_utils.normalize_keyvalues(onto_element)
            if onto_clean in name2cids[i]:
                ontology.remove(onto_element)
                #If there are several cluster with this name as TOPx, then we pick the bigger and we set the others as duplicates.
                #We pick only elements that were no tagged in other ways
                elements_filtered = [elem for elem in name2cids[i][onto_clean] if elem[CLASSIFICATION] == 'FP']
                elements_sorted = sorted(elements_filtered, key=lambda elem: elem[OCCURRENCES], reverse=True)
                elements_sorted[0][constants.NAME] = onto_element
                elements_sorted[0][CLASSIFICATION] = 'TP'
                tp += 1
                for duplicate in elements_sorted[1:]:
                    duplicate[constants.NAME] = onto_element
                    duplicate[CLASSIFICATION] = 'duplicated'
                    # We reduce computed positives: otherwise duplicated would be counted as false positives
                    computed_positives -= 1
    #Now we set as FN the elements in ontology that were not found in TOP1,2,3
    for elem in ontology:
        algo_output.append({constants.CLUSTER_ID: 'NA', constants.NAME: elem, CLASSIFICATION: 'FN'})

    metrics = experiment_utils.EvaluationMetrics(tp, expected_positives, computed_positives)
    return algo_output, metrics


def _safe_log(number):
    if number < 1:
        return -1
    else:
        return math.log(number)