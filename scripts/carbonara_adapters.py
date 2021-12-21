import bisect
import collections
import json
import os
import re

from tqdm import tqdm

from adapter import adapter_factory
from model import dataset, datamodel
from scripts.results_evaluation import ENTITY_ID
from adapter.output_and_gt_adapter import ATTRIBUTE_NAME, SOURCE_NAME, TARGET_ATTRIBUTE_ID, URL
from utils import io_utils, string_utils, bdsa_utils

from config.bdsa_config import _config_

# Constants in CARBONARA
TARGET_ATTRIBUTE_NAME = 'target_attribute_name'
PREDICTIONS_90 = 'predictions90'
CARBO_PROVENANCES = 'provenances'
CARBO_CLAIM = 'claims'
CARBO_INSTANCES = 'instances'
CARBO_RESOURCE_ID = 'resource_id'
CARBO_JSON_ID = 'json_id'
CARBO_JSON_NUMBER = 'json_number'
CARBO_SOURCE_ID = 'source_id'
CARBO_SOURCE_ATTRIBUTE_NAME = 'source_attribute_name'
CARBO_SOURCE_NAME = 'source_name'
CARBO_TARGET_ATTRIBUTE_IDS = 'target_attribute_ids'
CARBO_TARGET_ATTRIBUTE_ID = 'target_attribute_id'

DEFAULT_LINKAGE_TAG= 'DEFAULT'

class ImportedCarbonaraSource:
    def __init__(self, source_name, json_number2json_imported:dict):
        self.source_name = source_name
        self.number2imported = json_number2json_imported

def convert_carbonara_dataset_dir_to_bdsa(category:str, dataset_directory:str, json_files_path:str, entities_path:str):
    """
    Convert carbonara data to bdsa, also with linkage
    :param dataset_directory: directory of carbonara data
    :param json_files_path: location of json file path
    :param entities_path: location of entity file path
    :return:
    """

    # If linkage data are not present then  we do not import them, also we keep original URLs:
    linkage_data_present = json_files_path and entities_path
    if linkage_data_present:
        print('Import linkage data...')
        json_files = io_utils.import_json_file(json_files_path)
        entities = io_utils.import_json_file(entities_path)
        source2entity2page = _import_linkage_data(entities, json_files)
    else:
        source2entity2page = collections.defaultdict(dict)
    adapter_factory.spec_factory().persist_specifications_functional('carbo2bdsa',
        source_linkage_retriever=lambda source: source2entity2page[source.site],
        source_spec_retriever= lambda: _import_carbonara_data_generator_per_source(dataset_directory, category))

def convert_carbonara_dataset_with_expanded_linkage_clusters(category:str, dataset_directory:str, expanded_clusters_file:str):
    """
    Convert carbonara data to bdsa, also with linkage
    :param dataset_directory: directory of carbonara data
    :param json_files_path: location of json file path
    :param entities_path: location of entity file path
    :return:
    """

    # If linkage data are not present then  we do not import them, also we keep original URLs:
    if expanded_clusters_file:
        print('Import linkage data...')
        expanded = io_utils.import_json_file(expanded_clusters_file)
        source2entity2page = _import_expanded_clusters(expanded)
    adapter_factory.spec_factory().persist_specifications_functional('carbo2bdsa',
        source_linkage_retriever=lambda source: source2entity2page.get(source.site, {}),
        source_spec_retriever= lambda: _import_carbonara_data_generator_per_source(dataset_directory, category,
                        url_converter= lambda source, page: '%s//%s' % (source, page)))

def _import_expanded_clusters(entity2linkage):
    source2entity2url = collections.defaultdict(bdsa_utils.dd_set_generator)
    for entity, config2linkage in entity2linkage.items():
        for url in (config2linkage[PREDICTIONS_90] if PREDICTIONS_90 in config2linkage else []):
            source_name = url.split('//')[0]
            source2entity2url[source_name][entity].add(url)
    return {source: {entity:sorted(urls) for entity, urls in entity2url.items()} for source, entity2url in source2entity2url.items()}

def _clean_specifications(specification:dict) -> dict:
    """
    Remove unstructured data and merge lists
    :param carbo_source:
    :return:
    """
    if '__unstructured' in specification:
        del specification['__unstructured']
    # normalize lists
    for element in specification.keys():
        current = specification[element]
        if isinstance(current, list):
            specification[element] = '|'.join(current)
    return specification


def _import_carbonara_data_generator_per_source(directory:str, category:str,
                url_converter=lambda source_name, pg_number:_generate_page(source_name, pg_number).url) -> ImportedCarbonaraSource:
    """
    Import carbonara data, generator that yields data of each source
    :param directory:
    :return: ImportedCarbonaraSource
    """
    source_directories = io_utils.browse_directory_files(directory)
    for source_name, source_dir_path in tqdm(source_directories, desc='Converting source directories...'):
        json_url2imported = {}
        for json_file_name, json_path in io_utils.browse_directory_files(source_dir_path, lambda file_name: file_name.endswith('.json')):
            page_number = json_file_name.replace('.json','')
            page_specifications = _clean_specifications(io_utils.import_json_file(json_path))
            page_url = page_specifications['url'] if 'url' in page_specifications and not url_converter \
                else url_converter(source_name, page_number)
            page_specifications.pop('url', None)
            json_url2imported[page_url] = page_specifications
        yield datamodel.SourceSpecifications(source_name, category, json_url2imported)
        #yield ImportedCarbonaraSource(source_name, json_number2imported)


# LINKAGE DATA IMPORTER

def _import_linkage_data(entities:list, json_files:list):
    """
    Import record linkage data from carbonara entities.json file
    :param entities:
    :return:
    """
    json_id2page = _build_json_id2page(json_files)
    source2entity2url = collections.defaultdict(bdsa_utils.dd_list_generator)
    for entity in tqdm(entities, desc='Importing entities for linkage...'):
        for json_id in entity[CARBO_INSTANCES]:
            page = json_id2page[json_id]
            entity_id = entity[CARBO_RESOURCE_ID]
            bisect.insort(source2entity2url[page.source.site][entity_id], page.url)
    return source2entity2url

def _build_json_id2page(json_files:list):
    """
    Build JSON_ID2url map, using carbonara file json_files.JSON
    :param entities:
    :return:
    """
    json_id2page = {}
    for json_file in json_files:
        json_id = json_file[CARBO_RESOURCE_ID]
        json_number = json_file[CARBO_JSON_NUMBER]
        source_name = json_file[CARBO_SOURCE_NAME]
        json_id2page[json_id] = _generate_page(source_name, json_number)
    return json_id2page



def _generate_page(source_name:str, json_number, category='default'):
    """
    Generate URL page name from source and json file
    :param source_name:
    :param json_file_name:
    :return:
    """
    url = string_utils.url_normalizer("%s/%s" % (source_name, str(json_number)))
    return datamodel.page_factoryz(url, source_name)


### CONVERT GROUND TRUTH DATA

def convert_carbonara_schema_gt_file_to_bdsa(carbonara_gt_file):
    """
    Convert carbonara ground truth file into bdsa ground truth format

    :param carbonara_gt_file:
    :return:
    """
    source_attribute_gt = io_utils.import_json_file(carbonara_gt_file)
    output_gt = dataset.Dataset(['cluster_id_real', 'source','name'])
    for carbo_sa in tqdm(source_attribute_gt,'Importing ground truth...'):
        for target_attribute_id in carbo_sa[CARBO_TARGET_ATTRIBUTE_IDS]:
            output_gt.add_row({'cluster_id_real':target_attribute_id,
                               'source': carbo_sa['source_name'],
                               'name': carbo_sa['source_attribute_name']})
    output_gt.export_to_csv(_config_.get_output_dir(), 'ground_truth_converted_from_carbonara', True)

## BUILD INSTANCE-LEVEL GROUND TRUTH

TargetAttribute = collections.namedtuple('TargetAttribute', 'id name')

def convert_carbonara_provenance_gt_to_csv(cat:str, entities_gt_file:str, provenance_gt_file:str):
    print ('Import files...')
    entities = io_utils.import_json_file(entities_gt_file)
    provenances = io_utils.import_json_file(provenance_gt_file)
    ta2sa_url_entity = _build_ta2sa_url_entity(cat, provenances, entities)
    gt_output = dataset.Dataset([TARGET_ATTRIBUTE_ID, SOURCE_NAME, ATTRIBUTE_NAME, URL, ENTITY_ID])
    for ta, prov_list in sorted(ta2sa_url_entity.items(), key=lambda x: x[0]):
        for page_att_entity in sorted(prov_list):
            gt_output.add_row({TARGET_ATTRIBUTE_ID:ta.id, TARGET_ATTRIBUTE_NAME:ta.name, SOURCE_NAME: page_att_entity[0].source.site,
                               ATTRIBUTE_NAME:page_att_entity[1], URL:page_att_entity[0].url, ENTITY_ID:page_att_entity[2]})
    gt_output.export_to_csv(_config_.get_output_dir(), 'gt_instance_level_carbo', True)



def _build_provenance2page_att(category:str, provenances:list):
    """
    Internal method that builds a map of provenance ID 2 pair (url, sa)

    :param provenances: list of provenances in carbonara format
    :return:
    """
    prov2page_att = {}
    for prov in tqdm(provenances, desc='Converting provenances...'):
        page = _generate_page(prov[CARBO_SOURCE_NAME], prov[CARBO_JSON_NUMBER])
        prov2page_att[prov[CARBO_RESOURCE_ID]] = (page, prov[CARBO_SOURCE_ATTRIBUTE_NAME])
    return prov2page_att

def _build_ta2sa_url_entity(cat:str, carbonara_provenances:list, carbonara_entities:list):
    """
    Build a map target attribute --> provenances, given list of carbonara entities
    :param entities:
    :return:
    """
    prov2page_att = _build_provenance2page_att(cat, carbonara_provenances)
    ta2page_att_entity = collections.defaultdict(set)
    for carbo_entity in tqdm(carbonara_entities, desc='Converting entities...'):
        for claim in carbo_entity[CARBO_CLAIM]:
            ta = TargetAttribute(claim[CARBO_TARGET_ATTRIBUTE_ID], claim[TARGET_ATTRIBUTE_NAME])
            for prov_id in claim[CARBO_PROVENANCES]:
                page_att = prov2page_att[prov_id]
                ta2page_att_entity[ta].add((page_att[0], page_att[1], carbo_entity[CARBO_RESOURCE_ID]))
    return ta2page_att_entity


## OTHER METHODS


def _extract_model(json_data):
    """
    Extract model from ID
    :param json_data:
    :return:
    """
    title = json_data['<page title>']
    extraction = re.findall('(?!\w+\.)(?=[a-zA-Z0-9]*[a-zA-Z])(?=[a-zA-Z0-9]*[0-9])(?<!\.)[a-zA-Z0-9\-#]+', title)
    exclusion_regex = "[0-9]+(hz|in|cd|ms|v)|1080(p|i)|[0-9]+x[0-9]+"
    extraction_filtered = [val for val in extraction if len(val) > 2 and not re.fullmatch(exclusion_regex, val.lower())]
    return extraction_filtered

def cluster_builder_carbonara(directory):
    """
    Build clusters of JSONs sharing the same model, and outputs in json
    :param directory:
    :return:
    """
    res = _cluster_builder_carbonara_intern(directory)
    io_utils.output_json_file(res, 'model2json_id_list')

def _cluster_builder_carbonara_intern(directory, data_importer=_import_carbonara_data_generator_per_source, model_extractor=_extract_model):
    """
    Build clusters of JSONs sharing the same model
    :param directory:
    :return:
    >>> data_importer =  lambda x: [ImportedCarbonaraSource('s1', {1: 10, 2:10, 3: 20, 4:20, 5:30}), ImportedCarbonaraSource('s2', {15:20})]
    >>> model_extractor = lambda val: [val]
    >>> _cluster_builder_carbonara_intern('dummy', data_importer, model_extractor)
    {10: [1, 2], 20: [3, 4, 15]}
    """
    clusters = collections.defaultdict(set)
    for carbo_source in data_importer(directory):
        for json_id, json_data in carbo_source.number2imported.items():
            extracted_models = model_extractor(json_data)
            for model in extracted_models:
                clusters[model].add("%s//%s" % (carbo_source.source_name, json_id))
    cluster_serializable = {model:list(jid) for model, jid in clusters.items() if len(jid) > 1}
    return cluster_serializable