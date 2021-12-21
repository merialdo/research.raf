# coding=utf-8
import collections
import itertools
import json
import os
import re
import shutil

from tqdm import tqdm

import utils.io_utils
from adapter import abstract_specifications_adapter, adapter_factory
from config import constants
from config.bdsa_config import _config_
from model import bdsa_data_transformed, dataset
from model.datamodel import SourceSpecifications
from utils import string_utils, stats_utils, io_utils, bdsa_utils, tokenize_utils
from nltk import word_tokenize, ngrams

from utils.blocking_graph import MetablockingGraph

BAD_TEXT = ['click here', 'please', 'select']

def build_filtered_subset(min_att_per_page=3, min_page_per_att=3, min_page_per_source=3, min_att_per_source=3):
    """
    Filter subset keeping only pages with at least n attributes, only attributes in at least n pages,
    only sources with at least n pages and n attributes
    
    :param min_att_per_page: 
    :param min_page_per_att: 
    :param min_page_per_source: 
    :param min_att_per_source: 
    :return: 
    """

    #valid elements before pruning sources
    total_valid_attributes = 0
    total_valid_pages = 0

    #output
    sites2category2page2att2value = collections.defaultdict(dict)
    spec_adapter = adapter_factory.spec_factory()
    linkage_adapter = adapter_factory.linkage_factory(normalize_url=False)
    source2pid2urls = {}
    for source in spec_adapter.specifications_generator(normalize_data=False):

        #remove attributes with bad names or values (>50 chars, <=1 char...)
        _filter_bad_attributes(source.pages)
        #print('bad attributes filtered')

        #remove pages with few attributes and attributes in few pages, repeat until convergence
        new_pages, valid_atts = _apply_page_attribute_filtering_until_convergence(min_att_per_page,
                                                                                           min_page_per_att, source.pages)
        total_valid_attributes += len(valid_atts)
        total_valid_pages += len(new_pages)

        ## remove source if it has few valid pages or few valid attributes
        if len(new_pages) >= min_page_per_source and len(valid_atts) >= min_att_per_source:
            pid2urls = collections.defaultdict(set)
            for url in new_pages.keys():
                for pid in linkage_adapter.ids_by_url(url, source.site, source.category):
                    pid2urls[pid].add(url)
            source2pid2urls[source] = {pid: sorted(urls) for pid, urls in pid2urls.items()}
            sites2category2page2att2value[source.site][source.category] = new_pages
        #print('end with source %s'%(str(source)))

    #persist filtered sources
    spec_adapter.persist_specifications(sites2category2page2att2value, lambda source: source2pid2urls[source])
    print("valid attributes: "+str(total_valid_attributes))
    print("valid pages: "+str(total_valid_pages))

def build_subset_head_attributes():
    """
    Build a dataset with only head attributes for each source
    :return:
    """
    spec_gen = adapter_factory.spec_factory()
    sites2category2page2att2value = collections.defaultdict(dict)

    for source in spec_gen.specifications_generator():
        att2occs = collections.Counter()
        for url, specs in source.pages.items():
            att2occs.update(specs.keys())

        # Here we put 2 lists of HEAD and TAIL attributes
        ht_atts = collections.defaultdict(set)
        stats_utils.compute_head_tail(att2occs.keys(), lambda x: att2occs[x], lambda x, ht: ht_atts[ht].add(x))
        new_pages = {}
        for url, specs in source.pages.items():
            new_pages[url] = {k: v for k, v in specs.items() if k in ht_atts[stats_utils.HEAD]}
        sites2category2page2att2value[source.site][source.category] = new_pages
    spec_gen.persist_specifications(sites2category2page2att2value)

def fix_extraction_error_on_dataset():
    """
    Fix extraction error with names in attribute value (cf _fix_extraction_error_on_page)
    :return:
    """
    spec_gen = adapter_factory.spec_factory()
    sites2category2page2att2value = collections.defaultdict(dict)

    for source in spec_gen.specifications_generator(False):
        attnames = set()
        new_pages = {}
        # collect attribute names
        for url, specs in source.pages.items():
            attnames.update(tuple(word_tokenize(x.lower())) for x in specs.keys())
        for url, specs in source.pages.items():
            new_specs = _fix_extraction_error_on_page(attnames, specs)
            new_pages[url] = new_specs
        sites2category2page2att2value[source.site][source.category] = new_pages
    spec_gen.persist_specifications(sites2category2page2att2value)


def _fix_extraction_error_on_page(attnames, page_specs):
    """
    Fix extraction error shown in doctest
    :param attnames: a list of attribute names already found in pages of that source
    :param page_specs: original specifications extracted from that page
    :return: new specifications with error fixed
    >>> page_specs = {'brand':'canon', 'Resolution':'16 mpx', 'model':'D40 compact,Bundled items:USB cable red eye reduction: yes', \
    'memory':'16MB battery: ni-mh\\nfocal length: 12mm', 'auto focus':'yes: 16 points'}
    >>> attnames_list = ['brand','model','resolution', 'bundled items', 'auto focus','memory','battery','focal length', 'red eye reduction']
    >>> attnames = set(tuple(word_tokenize(x)) for x in attnames_list)
    >>> sorted(_fix_extraction_error_on_page(attnames, page_specs).items())
    [('auto focus', 'yes : 16 points'), ('battery', 'ni-mh'), ('brand', 'canon'), ('bundled items', 'USB cable'), \
('focal length', '12mm'), ('memory', '16MB'), ('model', 'D40 compact ,'), ('red eye reduction', 'yes'), ('resolution', '16 mpx')]
    """
    new_page_specs = {}
    for key, value in page_specs.items():
        value = value.replace(':', ' : ')
        tokens = word_tokenize(value)
        colon_pos = (pos for pos, value in enumerate(tokens) if value == ':')

        prev_colon_position = -1  # Position of previous colon found (we look for  attribute name before the current colon found and after previous one)
        new_value_beginning_position = 0  # If we extracted parte of the value then we move forward
        # for each colon we found we look for n-gram at left
        for pos in colon_pos:
            for ngram_length in range(pos - prev_colon_position - 2, 0, -1):
                candidate_att_name = tuple(v.lower() for v in tokens[pos - ngram_length:pos])
                if candidate_att_name in attnames:
                    new_page_specs[key.lower()] = ' '.join(tokens[new_value_beginning_position:pos - ngram_length])
                    new_value_beginning_position = pos + 1
                    key = ' '.join(candidate_att_name)
                    break
            prev_colon_position = pos
        new_page_specs[key.lower()] = ' '.join(tokens[new_value_beginning_position:])
    return new_page_specs


def _good_attribute(k, v):
    """
    Return true if the attribute instance is good, according to some patterns (number of characters, regex...)
    :param k: 
    :param v: 
    :return: 
    """
    mod_k = string_utils.folding_using_regex(k)

    res = len(k) < 50 and len(mod_k) >= 2 and re.match('^(\$|\€|\£)[0-9.,]+$', k) is None
    if res:
        tokens = tokenize_utils.value2token_set(mod_k)
        for elem in BAD_TEXT:
            elem_t = tokenize_utils.value2token_set(elem)
            res = res and not(elem_t.issubset(tokens))

    return res

def _filter_bad_attributes(pages):
    """
    Remove bad attributes from pages, according to some patterns  (see __good_attribute)
    :param pages: 
    :return: 
    """
    for url in pages:
        pages[url] = {k:v for k,v in pages[url].items() if _good_attribute(k, v)}

def _apply_page_attribute_filtering_until_convergence(min_att_per_page, min_page_per_att, pages):
    """
    Apply __filter_valid_attributes_page iteratively until convergence (until no attribute is removed)
    :param min_att_per_page: 
    :param min_page_per_att: 
    :param source: 
    :return: 
    """
    converges = False
    new_pages = pages
    while not converges:
        valid_atts, valid_pages = _filter_valid_attributes_page(min_att_per_page, min_page_per_att, new_pages)
        converges = len(valid_pages) == len(new_pages)
        new_pages = valid_pages
    return new_pages, valid_atts


def _filter_valid_attributes_page(min_att_per_page, min_page_per_att, pages):
    """
    Return valid attributes and pages in source, according to min_att_per_page and min_page_per_att
    :param min_att_per_page: 
    :param min_page_per_att: 
    :param pages: 
    :return: 
    """
    valid_pages = {}
    attributes = collections.defaultdict(int)
    for specs in pages.values():
        for att in specs:
            attributes[string_utils.normalize_keyvalues(att)] += 1

    ## filter attributes present in at least x pages
    valid_attributes = [k for k, v in attributes.items() if v >= min_page_per_att]
    ## remove non-valid attributes and re-filter pages with at least x valid attributes
    for url, specs in pages.items():
        specs_to_keep = {key: value for key, value in specs.items()
                         if string_utils.normalize_keyvalues(key) in valid_attributes}

        if len(specs_to_keep) >= min_att_per_page:
            valid_pages[url] = specs_to_keep
    return valid_attributes, valid_pages


def build_random_subset(number_of_sources=30, category=None):
    #output
    sites2category2page2att2value = collections.defaultdict(dict)
    specs = [source for source in abstract_specifications_adapter.specifications_generator(False, False, False)
             if source.category == category or not(category)]

    fixed_criterias = [] if category else [stats_utils.NamedLambda(lambda source: source.category, "Source category")]
    sample_data = stats_utils.sample_dataset(3, number_of_sources, specs,
                               [stats_utils.NamedLambda(lambda source: len(source.pages), "Number of pages")], fixed_criterias)
    for source in sample_data:
        sites2category2page2att2value[source.site][source.category] = source.pages
    abstract_specifications_adapter.persist_specifications(sites2category2page2att2value)



def get_clean_dataset():
    """
    Build a clean dataset, merging duplicates and removing empty pages
    :param input_dir: 
    :return: 
    """

    #output
    sites2category2page2att2value = collections.defaultdict(dict)
    #map of url2source2page of all sources
    url2source_specs = {}
    spec_adapter = adapter_factory.spec_factory()
    for source in spec_adapter.specifications_generator():

        #map site2url for current source
        site = source.site.replace('www.', '')
        #an existing source with an equivalent name (www.ebay.com vs ebay.com) may already be in output,
        # so we retrieve it.
        source_pages = sites2category2page2att2value[site].setdefault(source.category, {})

        #clear source
        for url, specs in source.pages.items():
            if len(specs) >= 1: #keep only if it has at least 1 attribute
                url = string_utils.url_normalizer(url)
                if url in url2source_specs:
                    (the_source, spec2) = url2source_specs[url]
                    print('Duplicate URL in different source: %s, %s, merging specs'%(str(the_source), url))
                    spec2.update(specs)
                else:
                    source_pages[url] = specs
                    url2source_specs[url] = (source, specs)

        _remove_empty_sources_or_sites(site, sites2category2page2att2value, source, source_pages)

    spec_adapter.persist_specifications(sites2category2page2att2value)


def _remove_empty_sources_or_sites(site, sites2category2page2att2value, source, source_pages):
    """
    Remove sources with 0 pages or sites with 0 categories (after merging)
    :param site: 
    :param sites2category2page2att2value: 
    :param source: 
    :param source_pages: 
    :return: 
    """
    if len(source_pages) == 0:
        print('removing source %s' % (str(source)))
        del sites2category2page2att2value[site][source.category]
        if len(sites2category2page2att2value[site]) == 0:
            print('removing site %s' % (site))
            del sites2category2page2att2value[site]

def merge_cat_comm_linkage():
    """
    Build clean data for record linkage: union of community and category linkage file.
    URLS must be linked if they pertain in same id, community and category in both linkages,
    and ID in both file must be the same.
    
    :return: 
    """
    #FIXME use generic path
    with open('/home/federico/BDSA/data/dexa/id2comm2urls.json', 'r') as infile:
        id2comm = json.load(infile)
    print ('Imported id2comm')
    with open('/home/federico/BDSA/data/dexa/id2category2urls.json', 'r') as infile:
        id2cat = json.load(infile)
    print('Imported id2cat')
    result = _merge_cat_comm_intern(id2cat, id2comm)
    io_utils.output_json_file(result, 'category2id2comm2url')


def _merge_cat_comm_intern(id2cat, id2comm):
    """
    Internal method for _merge_cat_comm_intern (takes and return objects)
    :param id2cat: 
    :param id2comm: 
    :return:
    >>> id2cat = {1: {'camera': ['/1','/2','/3'], 'tv':['/4', '/5']}, 2: {'camera':['/6']},\
                3: {'tv': ['/7', '/8']}}
    >>> id2comm = {1: {'c1': ['/2','/1'], 'c2': ['/3', '/4']}, 10:{'c1':['/5', '/6']}, \
                3: {'c5':['/7', '/8','/9']}}
    >>> _merge_cat_comm_intern(id2cat, id2comm)
    {'camera': {1: {'c1': ['/1', '/2'], 'c2': ['/3']}}, 'tv': {1: {'c2': ['/4']},\
 3: {'c5': ['/7', '/8']}}}
    """
    output = collections.defaultdict(bdsa_utils.dd2_set_generator)
    url2idcomm = {}
    for pid, comm2url in tqdm(id2comm.items(), desc='Build inverse url2idcomm'):
        for comm, urls in comm2url.items():
            for url in urls:
                url2idcomm[url] = (pid, comm)
    for pid, cat2url in tqdm(id2cat.items(), desc='Merge elements'):
        for cat, urls in cat2url.items():
            for url in urls:
                idcomm = url2idcomm.get(url, None)
                if idcomm is not None and idcomm[0] == pid:
                    output[cat][idcomm[0]][idcomm[1]].add(url)
    result = {cat: {pid: {comm: sorted(urls) for comm, urls in comm2.items()} for pid, comm2 in id2.items()}
              for cat, id2 in output.items()}
    return result

def filter_record_linkage(directory):
    full_linkage = collections.defaultdict(int)

    # First step: check size of each ID
    for source_dir in io_utils.browse_directory_files(
            directory, filter=lambda file: not(file.endswith('txt'))):
        for linkage_file in io_utils.browse_directory_files(
                source_dir.path, filter=lambda file: file.endswith('linkage.json')):
            id2urls = io_utils.import_json_file(linkage_file.path)
            for id, urls in id2urls.items():
                full_linkage[id] += len(urls)

    # Second step: filtering
    good_ids = set(pid for pid, nb in full_linkage.items() if nb >= 2)
    for source_dir in io_utils.browse_directory_files(
            directory, filter=lambda file: not(file.endswith('txt'))):
        for linkage_file in io_utils.browse_directory_files(
                source_dir.path, filter=lambda file: file.endswith('linkage.json')):
            id2urls = io_utils.import_json_file(linkage_file.path)
            id2urls_filtered = {id: urls for id, urls in id2urls.items() if id in good_ids}
            shutil.move(linkage_file.path, linkage_file.path.replace('linkage', 'linkage_old'))
            io_utils.output_json_file(id2urls_filtered, linkage_file.name.replace('.json',''), source_dir.path,
                                      False, True)

def filter_record_linkage_dexter_file(min_urls_in_specifications=2, min_id_size=0, max_id_size=float("inf")):
    """
    Filter record linkage providing only URLs inside current specifications
    :return: 
    """

    ##retrieve all URLs in data
    all_specs = adapter_factory.spec_factory().specifications_generator()
    spec_urls = set()
    for source in all_specs:
        spec_urls.update([string_utils.url_normalizer(url) for url in source.pages.keys()])

    del all_specs
    with open(_config_.get_linkage_dexter(), 'r') as infile:
        current_linkage = json.load(infile)
        output = _filter_record_linkage_intern(current_linkage, max_id_size, min_id_size, min_urls_in_specifications,
                                               spec_urls)
    utils.io_utils.output_json_file(output, 'filtered_cat2id2comm2urls')


def _filter_record_linkage_intern(current_linkage, max_id_size, min_id_size, min_urls_in_specifications, spec_urls):
    """
    
    :param current_linkage: 
    :param max_id_size: 
    :param min_id_size: 
    :param min_urls_in_specifications: 
    :param spec_urls: must be pre-normalized 
    :return:
     
     >>> current_linkage = {'camera': {'123': {'c1': ['/1','/2'], 'c3': ['/4', '/5']}, \
     '12': {'c4':['/6', '/7']}},'tv': {'1000': {'c7': ['/5'], 'c8': ['/11', '/12']}}}
     >>> spec_urls = ['/1', '/2', '/4', '/6','/7', '/11', '/12']
     >>> spec_urls_norm = set(string_utils.url_normalizer(url) for url in spec_urls)
     >>> _filter_record_linkage_intern(current_linkage, 50, 3, 2, spec_urls_norm)
     {'camera': {'123': {'c1': ['http:///1', 'http:///2']}},\
 'tv': {'1000': {'c8': ['http:///11', 'http:///12']}}}
    """
    output = {}
    pbar = tqdm(total=sum(len(pids) for pids in current_linkage.values()), desc='Retrieving IDS...')
    for cat, id2comms2urls in current_linkage.items():
        catdata = collections.defaultdict(dict)
        for pid, comm2urls in id2comms2urls.items():
            pbar.update(1)
            if len(pid) >= min_id_size and len(pid) <= max_id_size:
                for comm, urls in comm2urls.items():
                    good_urls = set([string_utils.url_normalizer(url) for url in urls]) & spec_urls
                    if len(good_urls) >= min_urls_in_specifications:
                        catdata[pid][comm] = list(good_urls)
        output[cat] = {pid: {comm: sorted(urls) for comm, urls in comm2urls.items()} for pid, comm2urls \
                       in catdata.items()}
    return output


