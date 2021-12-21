import collections
import json
import os
import pathlib
from sys import argv
from urllib.parse import urlparse

import pandas
import tld as tld
from tqdm import tqdm

from adapter import adapter_factory
from model import datamodel
from utils import io_utils, bdsa_utils

WDC_PRODUCT_ID = 'cluster_id'

WDC_ID = 'id'

KEY_VALUE_PAIRS = 'keyValuePairs'
WDC_CATEGORY = 'category'

def convert_wdc_file(input_file, id_url_mapping):
    specifications = io_utils.import_json_file(input_file)
    id2site = io_utils.build_dict_from_csv(
        id_url_mapping, 'id', lambda rec:tld.get_tld(rec['url'], as_object=True).fld)
    site2cat2specs = collections.defaultdict(bdsa_utils.dd_dict_generator)
    source2pid2urls = collections.defaultdict(bdsa_utils.dd_list_generator)
    for spec in specifications:
        if spec[WDC_CATEGORY] is not None and spec[KEY_VALUE_PAIRS] is not None:
            site = id2site[spec[WDC_ID]]
            url = _url_generator(site, spec)
            source = datamodel.SourceSpecifications(site, spec[WDC_CATEGORY], None)
            source2pid2urls[source][spec[WDC_PRODUCT_ID]].append(url)
            site2cat2specs[site][spec[WDC_CATEGORY]][url] = spec[KEY_VALUE_PAIRS]
    adapter_factory.spec_factory().persist_specifications_functional(
        'wdc', lambda source: source2pid2urls[source],
        lambda: convert_site2cat2specs_to_source(site2cat2specs))


def _url_generator(site, wdc_spec):
    return '%s//%d' % (site, wdc_spec[WDC_ID])


def convert_site2cat2specs_to_source(site2cat2specs):
    for site, cat2specs in site2cat2specs.items():
        for cat, specs in cat2specs.items():
            yield datamodel.SourceSpecifications(site, cat, specs)

def convert_wdc_file_optimized(input_file, id_url_mapping):
    """
    Convert WDC to RAF format, optimized for big files to avoid memory leak.
    Expects input_file as 1 json per line (not a well formed json)
    :param input_file:
    :param id_url_mapping:
    :return:
    """
    id2site = io_utils.build_dict_from_csv(
        id_url_mapping, WDC_ID, lambda rec: _get_site_domain(rec))
    output_dir = io_utils.build_directory_output('wdc_output')
    print('importing file...')
    for line in io_utils.import_generic_file_per_line(input_file, True):
        if "keyValuePairs\":{" in line and '"category":"' in line:
            spec_wdc = json.loads(line)
            site = id2site[str(spec_wdc[WDC_ID])]
            product_id = spec_wdc[WDC_PRODUCT_ID]
            spec_raf = {'url': _url_generator(site, spec_wdc), 'spec': spec_wdc[KEY_VALUE_PAIRS]}
            category = spec_wdc[WDC_CATEGORY]
            _add_to_file(output_dir, site, category, product_id, spec_raf)
    #fix_output_dir(output_dir)


def fix_output_dir(output_dir):
    """Fix output dir, convert linkage files from CSV to JSON, and fix JSON files """
    for source_dir in tqdm(io_utils.browse_directory_files(output_dir), desc='Converting dirs...'):
        for linkage_file_path_name in io_utils.browse_directory_files(source_dir[1], lambda file: file.endswith('linkage.csv')):
            linkage_file_fullpath = linkage_file_path_name[1]
            linkage_filename_json = linkage_file_path_name[0].replace('.csv', '')
            linkage_source = io_utils.build_dict_from_csv(linkage_file_fullpath, WDC_PRODUCT_ID, lambda val: val['url'],
                                                          multi=True)
            io_utils.output_json_file(linkage_source, linkage_filename_json, source_dir[1], timestamp=False)
            os.remove(linkage_file_fullpath)
        for linkage_file_path_name in io_utils.browse_directory_files(source_dir[1], lambda file: file.endswith('spec.json')):
            output_json = []
            first = True # verify if file is already fixed
            already_json = False
            for line in io_utils.import_generic_file_per_line(linkage_file_path_name[1], True):
                if first and line.startswith('['):
                    already_json  = True
                    break
                first = False
                output_json.append(json.loads(line))
            if not already_json:
                io_utils.output_json_file(output_json, linkage_file_path_name[0].replace('.json', ''), source_dir[1], timestamp=False)


def _get_site_domain(rec):
    try:
        res = tld.get_tld(rec['url'], as_object=True).fld
    except tld.exceptions.TldDomainNotFound:
        res = urlparse(rec['url']).netloc
    return res


def _add_to_file(output_dir, site, category, product_id, spec_raf):
    """
    Add the provided spec to the specification file of the source
    :param site:
    :param category:
    :param spec_raf:
    :return:
    """
    source_dir = os.path.join(output_dir, site)
    pathlib.Path(source_dir).mkdir(parents=True, exist_ok=True)
    spec_filename = '%s_spec.json' % category
    spec_filepath = os.path.join(source_dir, spec_filename)
    linkage_filename = '%s_linkage.csv' % category
    io_utils.append_generic_file(spec_filepath, json.dumps(spec_raf, sort_keys=True))
    io_utils.append_csv_file([WDC_PRODUCT_ID, 'url'], [{
        WDC_PRODUCT_ID: product_id, 'url': spec_raf['url']}], linkage_filename, source_dir)

if __name__ == '__main__':
    if argv[1] == 'fix':
        fix_output_dir(argv[2])
    else:
        convert_wdc_file_optimized(argv[1], argv[2])