import glob
import json
import os

import collections

import itertools
import shutil

from tqdm import tqdm

from adapter import adapter_factory
from config.bdsa_config import _config_
from model import dataset, datamodel
from model.clusters import FreeClusterRules
from pipeline import cluster_utils
from pipeline.cluster_utils import WeightedEdge
from utils import string_utils, bdsa_utils, io_utils


## This file manages the new clean linkage produced by ID extractor work.
## The output is a json file with category2id2community2files

# BE CAREFUL to correctly set the linkage configuration file

def build_linkage(clean=False, postfix='linkage_dirty'):
    """
    Builds the linkage data in format required by BDSA ([category]_linkage.json file in site's directory  
    :return: 
    """
    with open(_config_.get_linkage_dexter_combined_clean(), 'r') as linkage_file:
        cat2id2comm2urls = json.load(linkage_file)
        url2idcomm = collections.defaultdict(set)
        for id2comm2urls in cat2id2comm2urls.values():
            for pid, comm2urls in tqdm(id2comm2urls.items(), desc='Analyze id of this category...'):
                for comm, urls in comm2urls.items():
                    for url in urls:
                        output_pid = '%s___%s'%(pid, comm) if clean else pid
                        url2idcomm[string_utils.url_normalizer(url)].add(output_pid)

        for source in tqdm(adapter_factory.spec_factory().specifications_generator(), desc='Building linkage per source'):
            id2url_source = collections.defaultdict(list)
            for url in source.pages:
                if string_utils.url_normalizer(url) in url2idcomm:
                    idcomms = url2idcomm[url]
                    for idcomm in idcomms:
                        id2url_source[idcomm].append(url)
            with open(os.path.join(_config_.get_specifications(), source.site, '%s_%s.json'%(source.category, postfix)), 'w') as json_output:
                json.dump(id2url_source, json_output, indent=4)

def compute_nb_linkages():
    with open(_config_.get_linkage_dexter(), 'r') as linkage_file:
        cat2id2comm2urls = json.load(linkage_file)
        total_linkages = set()
        for id2comm2urls in cat2id2comm2urls.values():
            for comm2urls in id2comm2urls.values():
                for urls in comm2urls.values():
                    urls.sort()
                    total_linkages.update((a[0], a[1]) for a in itertools.combinations(urls, 2))
    print ("Nb of output elements: %d"%len(total_linkages))


def get_urls_from_other_dataset(dataset_without_urls, dataset_with_urls_dir):
    for source_dir in tqdm(os.listdir(dataset_without_urls), desc='Analyzing each source dir...'):
        for page_json_file in os.listdir(os.path.join(dataset_without_urls, source_dir)):
            #We open the page WITH url
            json_with_url_path = os.path.join(dataset_with_urls_dir, source_dir, page_json_file)
            json_without_url_path = os.path.join(dataset_without_urls, source_dir, page_json_file)
            if not os.path.exists(json_with_url_path):
                print("***ERROR***: file %s does not exist, skipped" % json_with_url_path)
            else:
                with open(json_with_url_path, 'r') as json_with_url:
                    source_data = json.load(json_with_url)
                    url = source_data['url']
                with open(json_without_url_path, 'r') as json_without_url:
                    source_data = json.load(json_without_url)
                    source_data['url'] = url
                with open(json_without_url_path, 'w') as json_without_url:
                    json.dump(source_data, json_without_url, indent=2)

def copy_linkage_from_dataset(dir1, dir2):
    """
    Copy linkage from one dataset to the other
    :param dir1:
    :param dir2:
    :return:
    """
    for asite in os.listdir(dir1):
        site_path = os.path.join(dir1, asite)
        dest_path = os.path.join(dir2, asite)
        if os.path.isdir(site_path) and os.path.exists(dest_path):
            for file in glob.glob(os.path.join(dir1, site_path, '*linkage*')):
                shutil.copy(file, dest_path)

CsvData = collections.namedtuple('CsvData', ['pid','source','url', 'name', 'value', 'additional_id'])
def convert_dataset_csv():
    """
    Convert dataset into CSV
    :return: 
    """
    output = dataset.Dataset(CsvData._fields)

    linkage_adapter = adapter_factory.linkage_factory(_config_.get_linkage_suffix())
    dataset_gen = adapter_factory.spec_factory().specifications_generator()
    for source in dataset_gen:
        site = source.site
        for url, a2v in source.pages.items():
            pids = linkage_adapter.ids_by_url(url, site, source.category)
            one_id = False
            for pid in pids:
                for att, value in a2v.items():
                    data = CsvData(pid, site, url, att, value, one_id)
                    output.add_row(data._asdict())
                one_id = True
    output.export_to_csv(_config_.get_output_dir(), _config_.get_category()+'_data', True)

def convert_dataset_csv_pivoted():
    """
    Convert dataset into CSV
    :return:
    """

    dataset_gen = adapter_factory.spec_factory().specifications_generator()
    for source in dataset_gen:
        atts = set()
        site = source.site
        for url, a2v in source.pages.items():
            atts.update(a2v.keys())

        all_atts = ['id'] + list(atts - {'<page title>'})
        output = dataset.Dataset(all_atts)
        for url, a2v in source.pages.items():
            row = dict(a2v)
            row.update({'id': url})
            del row['<page title>']
            output.add_row(row)
        output.export_to_csv(_config_.get_output_dir(), site+'__'+_config_.get_category()+'_data', True)

def convert_linkage_pairs_to_bdsa_linkage(pairs_file:str, suffix:str):
    """
    Convert linkage from pairs and add to current dataset
    :param pairs_file: 
    :return: 
    """
    # Import files
    pairs = dataset.import_csv(pairs_file, ['left_spec_id', 'right_spec_id'])
    edges = []
    for row in tqdm(pairs.rows, desc='Convert rows to edges...'):
        edges.append(WeightedEdge(_convert_pageurl_to_page(row['left_spec_id']),
                                  _convert_pageurl_to_page(row['right_spec_id']), 1))

    pid2source2pages = collections.defaultdict(bdsa_utils.dd_set_generator)
    cluster_utils.partition_using_agglomerative(edges, pid2source2pages, FreeClusterRules())
    source2pid2urls = collections.defaultdict(bdsa_utils.dd_set_generator)
    for pid, source2pages in tqdm(pid2source2pages.items(), desc='Convert format...'):
        for source, pages in source2pages.items():
            source2pid2urls[source][pid].update(string_utils.url_normalizer(page.url) for page in pages)

    for source, pid2urls in tqdm(source2pid2urls.items(), desc='Cleaning linkage...'):
        spec_factory = adapter_factory.spec_factory()\
            .source_specifications(source.site, _config_.get_category())
        source_urls = spec_factory.pages.keys()
        for pid, urls in dict(pid2urls).items():
            pid2urls[pid] = {url for url in urls if url in source_urls}
    linkage_adapter = adapter_factory.linkage_factory(suffix)
    linkage_adapter.persist_linkage_data(source2pid2urls)




def _convert_pageurl_to_page(url):
    return datamodel.page_factoryz(url, url.split('//')[0], _config_.get_category())


if __name__ == '__main__':
    convert_dataset_csv_pivoted()
