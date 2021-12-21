import collections
import itertools

from config import constants
from config.bdsa_config import _config_
from model import dataset, datamodel
from model.datamodel import convert_provenance, Provenance
from scripts import script_commons
from utils import string_utils, io_utils, bdsa_utils

ADDITIONAL_ENTITIES = 'additional_entities'

MAIN_ENTITY = 'main_entity'

"""
Import and transform output file & ground truth. 
"""

### Import cluster detail file ###

class ClusterDetailOutput:
    """
    Output detail file
    """

    def __init__(self, _sa2comp_cluster={}, _sa2other_data={}):
        self.sa2clusters = dict(_sa2comp_cluster)
        self.sa2other_data = dict(_sa2other_data)

    def convert_to_csv(self, filename):
        specific_headers = ['source', 'name', 'cluster_id']
        output = dataset.Dataset(specific_headers)
        for sa, clusters in self.sa2clusters.items():
            for cl in clusters:
                row = {'source': sa.source.site, 'name':sa.name, 'cluster_id': cl}
                if self.sa2other_data and sa in self.sa2other_data:
                    row.update(self.sa2other_data[sa])
                output.add_row(row)
        output.export_to_csv(_config_.get_output_dir(), filename, False)

    def add_sa(self, sa, comp_cluster, other_data=None):
        self.sa2clusters[sa] = comp_cluster
        if other_data:
            self.sa2other_data[sa] = other_data


def import_cluster_detail_csv(cluster_att_name, file_path, extract_all_columns=False, category_input=None) -> ClusterDetailOutput:
    """
    Import an external file with cluster details (i.e. 1 row per attribute+cluster to which it belong).

    Works for output & ground truth.
    Note that all generated attributes are converted into original ones, and we will provide topX of original ones.
    :param cluster_att_name:
    :param file_name:
    :return:
    """
    category = category_input or _config_.get_category()
    specific_headers = ['source', 'name', cluster_att_name]
    headers = None if extract_all_columns else specific_headers
    cluster_detail_file = dataset.import_csv(file_path, headers)
    cdo = _import_detail_csv_intern(cluster_att_name, cluster_detail_file.rows, extract_all_columns,
                                                             specific_headers, category)

    return cdo

def _import_detail_csv_intern(cluster_att_name, rows: list, extract_all_columns,
                              specific_headers, category='dummy') -> ClusterDetailOutput:
    """
    Internal testable method for _import_cluster_detail_csv
    :param cluster_att_name:
    :param cluster_detail_file:
    :param extract_all_columns:
    :param specific_headers:
    :return:
    >>> rows = [{'source':'s1', 'cluster_id':10, 'name':'a1', 'full_name':'a1#-#1', 'top1': 'a1#1#top1', 'do_not_extract':12},\
        {'source':'s1', 'cluster_id':10, 'name':'a1', 'full_name':'a1', 'top1': 'a1#top', 'do_not_extract':15},\
        {'source':'s1', 'cluster_id':10, 'name':'a1', 'full_name':'a1#-#2', 'top1': 'a1##2#top', 'do_not_extract':16},\
        {'source':'s1', 'cluster_id':20, 'name':'a1', 'full_name':'a1#-#3', 'top1': 'a1##3#top', 'do_not_extract':4},\
        {'source':'s1', 'cluster_id':10, 'name':'b1', 'full_name':'b1', 'top1': 'b1#top', 'do_not_extract':4},\
        {'source':'s1', 'cluster_id':20, 'name':'b1', 'full_name':'b1#-#1', 'top1': 'b1##3#top', 'do_not_extract':4}]
    >>> res1, res2 = _import_detail_csv_intern('cluster_id', rows, True, ['source','name', 'cluster_id','do_not_extract'])
    >>> res1, res2
    ({SourceAttribute(source=SourceSpecifications(site='s1', category='dummy', pages=None), name='a1'): {10, 20}, \
SourceAttribute(source=SourceSpecifications(site='s1', category='dummy', pages=None), name='b1'): {10, 20}}, \
{SourceAttribute(source=SourceSpecifications(site='s1', category='dummy', pages=None), name='a1'): \
{'full_name': 'a1', 'top1': 'a1#top'}, \
SourceAttribute(source=SourceSpecifications(site='s1', category='dummy', pages=None), name='b1'): \
{'full_name': 'b1', 'top1': 'b1#top'}})

    """
    sa2cluster = collections.defaultdict(set)
    sa2other_columns = collections.defaultdict(dict)
    for row in rows:
        cid = row[cluster_att_name]
        nameatt = 'attribute_name' if 'attribute_name' in row else 'name'
        sa = datamodel.source_attribute_factory(category, row['source'], #.replace('www.', ''),
                                                string_utils.normalize_keyvalues(row[nameatt]))
        is_generated = constants.GENERATED_ATTS_SEPARATOR in row.get('full_name', '')

        if cid != '':
            sa2cluster[sa].add(cid)
        # If the attribute is original, then we collect all of its data
        if not is_generated and extract_all_columns and len(row.keys() - specific_headers) > 0:
            sa2other_columns[sa] = {col: value for col, value in row.items() if col not in specific_headers}
    return ClusterDetailOutput(dict(sa2cluster), dict(sa2other_columns))


PREDICATE_NAME = 'predicate_name'
CLUSTER_ID_REAL = 'cluster_id_real'
MULTIPLE = 'MULTIPLE'
VALUE = 'VALUE'


ATTRIBUTE_NAME = 'ATTRIBUTE_NAME'
SOURCE_NAME = 'SOURCE_NAME'
TARGET_ATTRIBUTE_ID = 'TARGET_ATTRIBUTE_ID'
URL = 'URL'

### Import ground truth instance-level AND output instance-level (knowledge graph PP)

class InstanceLevelClustering:
    """
    Instance-level clustering, GT or output of the algorithm
    """
    def __init__(self, prov2ta=collections.defaultdict(set), heterogeneous_sa=set()):
        """
        Provide target attributes with associated tuple name-url
        :param prov2ta:
        """
        self.prov2ta = prov2ta
        self.heterogeneous_sa = heterogeneous_sa

    def add_sa_full(self, cid, sa, sa2url_value):
        for url_value in sa2url_value.get(sa, []):
            prov = datamodel.provenance_factory(sa.get_site(), sa.get_category(), url_value.url, sa.name, url_value.value)
            self.prov2ta[prov].add(cid)

    def __repr__(self):
        return '\n'.join(str(x) for x in self.prov2ta.items()) + '\n' + ','.join(str(sa) for sa in self.heterogeneous_sa)

def build_il_clustering_from_gt(il_gt, category):
    """
    Import instance-level ground truth
    >>> rw = lambda name, url, ta: {SOURCE_NAME:'source', ATTRIBUTE_NAME:name, URL:url, VALUE: 'test', TARGET_ATTRIBUTE_ID: ta}
    >>> # Two attributes of a SA with same pair of TA: homogeneous
    >>> class input_gt: rows = [rw('a', '/1', 'brand'), rw('a', '/1', 'name'), rw('a', '/2', 'brand'), rw('a', '/2', 'name')]
    >>> input_gt.rows.extend([rw('b', '/1', 'batt chem'), rw('b', '/1', 'batt model'), rw('b', '/5', 'batt chem')])
    >>> build_il_clustering_from_gt(input_gt, 'dummy')
    (source#-#/1/a=test, {'name', 'brand'})
    (source#-#/2/a=test, {'name', 'brand'})
    (source#-#/1/b=test, {'batt model', 'batt chem'})
    (source#-#/5/b=test, {'batt chem'})
    source__dummy/b
    """
    prov2ta = collections.defaultdict(set)
    heterogeneous_sa = set()
    sa2prov2tas = collections.defaultdict(bdsa_utils.dd_set_generator)
    tot_prov_homonyms = 0
    for row in il_gt.rows:
        provenance = datamodel.provenance_factory(row[SOURCE_NAME], category, row[URL], row[ATTRIBUTE_NAME],
                                                  row.get(VALUE, ''))
        sa2prov2tas[provenance.sa][provenance].add(row[TARGET_ATTRIBUTE_ID])
        prov2ta[provenance].add(row[TARGET_ATTRIBUTE_ID])
    for sa, prov2tas in sa2prov2tas.items():
        # Quicker solution: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
        tas_groups = set(frozenset(tas_group) for tas_group in prov2tas.values())
        if len(tas_groups) > 1:
            heterogeneous_sa.add(sa)
            tot_prov_homonyms += len(prov2tas)
    print ('total homonyms: %d on %d (%f) ' % (tot_prov_homonyms, len(prov2ta), tot_prov_homonyms/len(prov2ta)))
    return InstanceLevelClustering(prov2ta, heterogeneous_sa)

def build_il_clustering_from_ikgpp(ikgpp_path, cat, filter=None):
    # import ikgpp_path
    ikgpp = io_utils.import_json_file(ikgpp_path)
    provenance2ta = collections.defaultdict(set)
    for entity, pred2provs in ikgpp.items():
        for ta_id, provs in pred2provs.items():
            for prov in provs:
                prov_obj = convert_provenance(prov, cat)
                if filter is None or filter(prov_obj):
                    provenance2ta[prov_obj].add(ta_id)
    return InstanceLevelClustering(provenance2ta, None)


### Transform ground truth

def convert_schema_level_to_instance_level_gt(only_linked=True, include_predicate_name=True, include_entity=True):
    """
    Convert schema-level GT to instance-level.
    Considers each occurrence of attribute automatically linked to all target attribute associated to its source attribute.
    A manual work of cleaning (removing wrong rows) should be done.
    Technically it is a product between attribute instances

    :param: only_linked if true, keep only URLS with at least 1 linkage
    :return:
    """
    cdo = import_cluster_detail_csv('cluster_id_real', _config_.get_ground_truth_path(),
                                            category_input=_config_.get_category())
    ground_truth_file = dataset.import_csv(_config_.get_ground_truth_path())
    cid2predicate_name = None
    if include_predicate_name:
        cid2predicate_name = _find_associations_cid_predicate_name(ground_truth_file)
    sa2url_value = script_commons.get_sa2urls(only_linked, True, True)
    ds = _combine_url_cids(cdo.sa2clusters, sa2url_value, cid2predicate_name)
    ds.export_to_csv(_config_.get_output_dir(), 'instance_level_gt', True)


def _find_associations_cid_predicate_name(ground_truth_file: dataset.Dataset):
    """
    Detect associations in GT file
    :return:
    >>> ground_truth_file = dataset.Dataset()
    >>> ground_truth_file.add_row({CLUSTER_ID_REAL:1, PREDICATE_NAME:'width'})
    >>> ground_truth_file.add_row({CLUSTER_ID_REAL:1, PREDICATE_NAME:'width'})
    >>> ground_truth_file.add_row({CLUSTER_ID_REAL:2, PREDICATE_NAME:'memory'})
    >>> _find_associations_cid_predicate_name(ground_truth_file)
    {1: 'width', 2: 'memory'}
    """
    # Re-importa tutto e prendi le associazioni ID - predicate_name
    cid2predicate_name = {}
    for row in ground_truth_file.rows:
        if row[CLUSTER_ID_REAL] not in cid2predicate_name:
            cid2predicate_name[row[CLUSTER_ID_REAL]] = row[PREDICATE_NAME]
        elif cid2predicate_name[row[CLUSTER_ID_REAL]] != row[PREDICATE_NAME]:
            raise Exception('Duplicate association for cid %s: %s and %s' % (row[CLUSTER_ID_REAL],
                                                                             cid2predicate_name[row[CLUSTER_ID_REAL]],
                                                                             row[PREDICATE_NAME]))
    return cid2predicate_name


def _combine_url_cids(sa2clusters, sa2url_value, cid2predicate_name:dict):
    """
    From input data build a candidate output instance data, with each attribute occurrence associated to each of its 
    possible clusters
    :param sa2clusters:
    :param sa2url_value:
    :param sa2other:
    :return:
    >>> sa2cl = {tsa('dim'): {1,2,3}, tsa('width'): {1}, tsa('flash'):{4}}
    >>> sa2urls = {tsa('dim'): {('a', 10),('b', 20),('c', 30)}, tsa('width'): {('e', 10), ('f', 50)}, tsa('flash'): {('g', 80)}}
    >>> cid2predicate_name = {1: 'width', 2: 'height', 3: 'depth', 4: 'flash'}
    >>> sorted(_combine_url_cids(sa2cl, sa2urls, cid2predicate_name).rows, key=lambda x: (x[ATTRIBUTE_NAME], x[TARGET_ATTRIBUTE_ID], x[URL]))
    [{'TARGET_ATTRIBUTE_ID': 1, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'dim', 'URL': 'a', 'VALUE': 10, 'MULTIPLE': True, 'predicate_name': 'width'},\
 {'TARGET_ATTRIBUTE_ID': 1, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'dim', 'URL': 'b', 'VALUE': 20, 'MULTIPLE': True, 'predicate_name': 'width'},\
 {'TARGET_ATTRIBUTE_ID': 1, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'dim', 'URL': 'c', 'VALUE': 30, 'MULTIPLE': True, 'predicate_name': 'width'},\
 {'TARGET_ATTRIBUTE_ID': 2, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'dim', 'URL': 'a', 'VALUE': 10, 'MULTIPLE': True, 'predicate_name': 'height'},\
 {'TARGET_ATTRIBUTE_ID': 2, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'dim', 'URL': 'b', 'VALUE': 20, 'MULTIPLE': True, 'predicate_name': 'height'},\
 {'TARGET_ATTRIBUTE_ID': 2, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'dim', 'URL': 'c', 'VALUE': 30, 'MULTIPLE': True, 'predicate_name': 'height'},\
 {'TARGET_ATTRIBUTE_ID': 3, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'dim', 'URL': 'a', 'VALUE': 10, 'MULTIPLE': True, 'predicate_name': 'depth'},\
 {'TARGET_ATTRIBUTE_ID': 3, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'dim', 'URL': 'b', 'VALUE': 20, 'MULTIPLE': True, 'predicate_name': 'depth'},\
 {'TARGET_ATTRIBUTE_ID': 3, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'dim', 'URL': 'c', 'VALUE': 30, 'MULTIPLE': True, 'predicate_name': 'depth'},\
 {'TARGET_ATTRIBUTE_ID': 4, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'flash', 'URL': 'g', 'VALUE': 80, 'MULTIPLE': False, 'predicate_name': 'flash'},\
 {'TARGET_ATTRIBUTE_ID': 1, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'width', 'URL': 'e', 'VALUE': 10, 'MULTIPLE': False, 'predicate_name': 'width'},\
 {'TARGET_ATTRIBUTE_ID': 1, 'SOURCE_NAME': 'test_site', 'ATTRIBUTE_NAME': 'width', 'URL': 'f', 'VALUE': 50, 'MULTIPLE': False, 'predicate_name': 'width'}]

    """
    ds = dataset.Dataset([TARGET_ATTRIBUTE_ID, SOURCE_NAME, ATTRIBUTE_NAME, URL, MAIN_ENTITY, ADDITIONAL_ENTITIES])
    for sa in sa2clusters.keys():
        multiple = len(sa2clusters[sa]) > 1
        for cid, url_value in itertools.product(sa2clusters[sa], sa2url_value[sa]):
            row = {TARGET_ATTRIBUTE_ID: cid, SOURCE_NAME: sa.source.site, ATTRIBUTE_NAME: sa.name,
                         URL: url_value.url, VALUE: url_value.value, MULTIPLE: multiple,
                   MAIN_ENTITY:url_value.pid, ADDITIONAL_ENTITIES:url_value.add_pid}
            if cid2predicate_name:
                row[PREDICATE_NAME] = cid2predicate_name[cid]
            ds.add_row(row)
    return ds
