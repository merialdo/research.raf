import collections
import itertools

import numpy
from tqdm import tqdm

import model.datamodel
from adapter import abstract_specifications_adapter, global_file_linkage, adapter_factory
from config.bdsa_config import _config_
from model import dataset, bdsa_data_transformed
from scripts.scripts_constants import *
from test.test_utils import tsp
from utils import stats_utils, io_utils, bdsa_utils, string_utils

### Compute some stats on 'schemas' of sources (union of its attribute names) and ratio of schemas used
from utils.blocking_graph import MetablockingGraph

MEDIAN_TAIL_PRODUCT_SIZE = 'Median tail product size'
MEDIAN_HEAD_PRODUCT_SIZE = 'Median head product size'

AVG_TAIL_PRODUCT_SIZE = 'Avg tail product size'
AVG_HEAD_PRODUCT_SIZE = 'Avg head product size'

TAIL_PRODUCTS = '#TAIL products'
HEAD_PRODUCT = '#HEAD product'

NB_DISTINCT_VALUES = '#Distinct values'
NB_SITES = '#Sites'

NB_ATTRIBUTE_INSTANCES = '#Attribute instances'
TOP5_ATT_PAGES = 'Top5 att - #pages'
TOP1_ATT_PAGES = 'Top1 att - #pages'

THRESHOLD_95 = 'threshold95'
THRESHOLD_85 = 'threshold85'
THRESHOLD_55 = 'threshold55'
THRESHOLD_10 = 'threshold10'

NON_ISOLATED_PAGES = '#Non-isolated pages'
AVERAGE_SCHEMA_USAGE_PERC = 'Avg Schema usage %'
MEDIAN_ATTRIBUTES_PER_PAGE = 'Median attributes per page'

MIN_PAGES_TO_COMPUTE = 2

SCHEMA_SIZE = 'Schema size'
THRESHOLDS = [1, 3, 7, 10, 30, 100, 300, 1000]

TEST_DATA = {
    model.datamodel.SourceSpecifications('fakesiteH', 'fakecategory',
                                         {
                                         'u1': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
                                         'u2': {'a': 1, 'b': 2, 'c': 1, 'e': 1},
                                         'u3': {'a': 1, 'b': 3, 'c': 1, 'f': 1},
                                         'u4': {'a': 1, 'b': 4, 'c': 1, 'g': 1},
                                         'u5': {'a': 1, 'b': 5, 'c': 1, 'h': 1},
                                         'u6': {'a': 1, 'b': 6, 'c': 1, 'i': 1},
                                         'u7': {'a': 1, 'b': 7, 'c': 1, 'j': 1}
                                     }),
    model.datamodel.SourceSpecifications('fakesiteT1', 'fakecategory2',
                                         {
                                         'u8': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
                                     }),
    model.datamodel.SourceSpecifications('fakesiteT2', 'fakecategory2',
                                         dict(u9={'a': 1, 'b': 1, 'c': 1, 'd': 1},
                                          u10={'a': 1, 'b': 1, 'c': 1, 'd': 1},
                                          u11={'a': 1, 'b': 1, 'c': 1, 'd': 1}))
}


def source_schema_analysis():
    """
    For each source, computes stats on ratio of schema used by each page, and ratio of pages that provide each attribute
    Moreover, extracts some notable pages/attribute (top/bottom according to ratio) to another dataset
    :return: 
    """
    source_dataset = dataset.Dataset([NAME, NUMBER_OF_PAGES, MEDIAN_ATTRIBUTES_PER_PAGE, SCHEMA_SIZE,
                                      AVERAGE_SCHEMA_USAGE_PERC, THRESHOLD_10, THRESHOLD_55, THRESHOLD_85,
                                      THRESHOLD_95, TOP1_ATT_PAGES, TOP5_ATT_PAGES])
    attribute_dataset = dataset.Dataset([NAME])
    for source in adapter_factory.spec_factory().specifications_generator():
        if len(source.pages) >= MIN_PAGES_TO_COMPUTE:
            source_label = source.site + "__" + source.category
            row = {NAME: source_label, NUMBER_OF_PAGES: len(source.pages)}

            # build schema
            schema = _build_schema_of_source(row, source)
            schema_size = len(schema)
            row[SCHEMA_SIZE] = schema_size
            row[MEDIAN_ATTRIBUTES_PER_PAGE] = int(numpy.median([len(spec) for spec in source.pages.values()]))

            # Note that we won't compute measure on ratio of attributes per page as they are
            # very similar or identical to page per attribute
            _compute_attribute_measures(schema, len(source.pages), row)
            _extract_topbottom_attributes(schema, len(source.pages), attribute_dataset, source_label)
            source_dataset.add_row(row)

    stats_utils.compute_head_tail_dataset(source_dataset, NUMBER_OF_PAGES)
    source_dataset.export_to_csv(_config_.get_output_dir(), 'source_schema', True)
    attribute_dataset.export_to_csv(_config_.get_output_dir(), 'attribute_schema', True)


### INTERN METHODS

def _extract_topbottom_attributes(schema, total_pages, attribute_dataset, source_label):
    attributes = sorted(schema, key=lambda att: schema[att], reverse=True)
    for att in attributes[:3]:
        attribute_dataset.add_row(
            {NAME: att, 'source': source_label, 'ratio': schema[att] / float(total_pages),
             NUMBER_OF_PAGES: schema[att], 'level': 'TOP'})
    for att in attributes[-3:]:
        attribute_dataset.add_row(
            {NAME: att, 'source': source_label, 'ratio': schema[att] / float(total_pages),
             NUMBER_OF_PAGES: schema[att], 'level': 'BOTTOM'})


def _build_schema_of_source(row, source):
    """
    
    :param row: 
    :param source: 
    :return: dictionary with attribute_name --> ratio of pages in which it is used 
    """
    # build schema
    schema_abs = collections.defaultdict(int)
    for url, specs in source.pages.items():
        for att_name in list(specs.keys()):
            schema_abs[att_name] += 1

    return schema_abs


def _compute_attribute_measures(schema, nb_total_pages, row):
    """
    Compute median nb pages per attribute, and some thresholds
    :param schema: 
    :param row: 
    :return: 
    """
    all_ratios = list(schema.values())
    row[AVERAGE_SCHEMA_USAGE_PERC] = int(round(numpy.mean(all_ratios) / nb_total_pages * 100))

    ## how many attributes are in at least 10,55, 85 or 95% of pages
    ratio10 = min(max(nb_total_pages * 0.1, 2), nb_total_pages - 1)
    row[THRESHOLD_10] = len([key for key in list(schema.keys()) if schema[key] >= ratio10])

    ratio55 = min(max(nb_total_pages * 0.55, 3), nb_total_pages - 1)
    row[THRESHOLD_55] = len([key for key in list(schema.keys()) if schema[key] >= ratio55])

    ratio85 = min(max(nb_total_pages * 0.85, 3), nb_total_pages - 1)
    row[THRESHOLD_85] = len([key for key in list(schema.keys()) if schema[key] >= ratio85])

    ratio95 = min(max(nb_total_pages * 0.95, 4), nb_total_pages - 1)
    row[THRESHOLD_95] = len([key for key in list(schema.keys()) if schema[key] >= ratio95])

    ## how many pages have the top 1 and top 5 attributes?
    ## TODO this should be merged with extract_topbottom_attributes
    sorted_attributes = sorted(list(schema.keys()), reverse=True, key=lambda key: schema[key])

    row[TOP1_ATT_PAGES] = schema[sorted_attributes[0]] if len(sorted_attributes) > 0 else 'NA'
    row[TOP5_ATT_PAGES] = schema[sorted_attributes[4]] if len(sorted_attributes) > 4 else 'NA'

GeneralStats = collections.namedtuple('GeneralStats', [NAME])

def compute_general_stats(linkage=False):
    """
    Compute some general stats on dataset
    :return: 
    """

    attributes = [CATEGORY, STATS_NB_SOURCES, NUMBER_OF_PAGES, NB_SOURCE_ATTRIBUTES, NB_ATTRIBUTE_INSTANCES]
    if linkage:
        attributes.extend([NON_ISOLATED_PAGES, AVG_HEAD_PRODUCT_SIZE, MEDIAN_HEAD_PRODUCT_SIZE,
                           AVG_TAIL_PRODUCT_SIZE,MEDIAN_TAIL_PRODUCT_SIZE,
                            HEAD_PRODUCT, TAIL_PRODUCTS])
    attributes.append(NB_DISTINCT_VALUES)
    general_stats_dataset = dataset.Dataset(attributes)
    sgen = adapter_factory.spec_factory().specifications_generator()
    linkage_adapter = adapter_factory.linkage_factory()

    all_sources = collections.defaultdict(int)
    all_pages = collections.defaultdict(int)
    all_attribute_values = collections.defaultdict(int)
    all_attribute_keys = collections.defaultdict(int)

    cat2pid2urls = collections.defaultdict(bdsa_utils.dd_set_generator)
    all_values = collections.defaultdict(set)


    categories = set()
    set_all_sites = set([])
    for source in sgen:
        cat = source.category
        categories.add(cat)
        set_all_attributes_keys = set([])
        all_sources[cat] += 1
        set_all_sites.add(source.site)
        for url, page_spec in source.pages.items():
            all_pages[cat] += 1
            all_values[cat].update(page_spec.values())
            set_all_attributes_keys.update(list(page_spec.keys()))
            nb_attributes = len(page_spec)
            all_attribute_values[cat] += nb_attributes
            if linkage:
                ids = linkage_adapter.ids_by_url(url, source.site, source.category)
                for pid in ids:
                    cat2pid2urls[cat][pid].add(url)
        all_attribute_keys[cat] += len(set_all_attributes_keys)
    if linkage:
        cat2non_isolated_pids2urls = {cat: {pid: urls for pid, urls in pid2urls.items() if len(urls) > 1} for cat, pid2urls in cat2pid2urls.items()}
        urls_in_linkage = {cat: len(set().union(*pids2urls.values()))
                           for cat, pids2urls in cat2non_isolated_pids2urls.items()}
    row = {CATEGORY: 'ALL', STATS_NB_SOURCES: sum(all_sources.values()), NUMBER_OF_PAGES: sum(all_pages.values()),
           NB_SOURCE_ATTRIBUTES: sum(all_attribute_keys.values()),
           NB_ATTRIBUTE_INSTANCES: sum(all_attribute_values.values()),
           NB_SITES:len(set_all_sites), NB_DISTINCT_VALUES: sum(len(x) for x in all_values.values())}
    general_stats_dataset.add_row(row)
    for cat in categories:
        row = {CATEGORY:cat, STATS_NB_SOURCES: all_sources[cat], NUMBER_OF_PAGES: all_pages[cat],
               NB_SOURCE_ATTRIBUTES: all_attribute_keys[cat],
               NB_ATTRIBUTE_INSTANCES: all_attribute_values[cat], NB_DISTINCT_VALUES: len(all_values[cat])}
        if linkage:
            row.update(_build_linkage_stats_category(urls_in_linkage[cat], cat2non_isolated_pids2urls[cat]))
        general_stats_dataset.add_row(row)

    general_stats_dataset.export_to_csv(_config_.get_output_dir(), 'general stats', True)


def _build_linkage_stats_category(non_isolated_urls, pid2urls):
    head_sizes = []
    tail_sizes = []
    stats_utils.compute_head_tail(pid2urls.keys(), lambda pid: len(pid2urls[pid]),
                                  lambda pid, ht: head_sizes.append(len(pid2urls[pid]))
                                  if ht == stats_utils.HEAD else tail_sizes.append(len(pid2urls[pid])))
    nb_head_clusters = len(head_sizes)
    nb_tail_clusters = len(tail_sizes)
    avg_head_size = int(round(numpy.average(head_sizes))) if nb_head_clusters > 0  else 'NA'
    avg_tail_size = int(round(numpy.average(tail_sizes))) if nb_tail_clusters > 0  else 'NA'
    median_head_size = int(numpy.median(head_sizes)) if nb_head_clusters > 0  else 'NA'
    median_tail_size = int(numpy.median(tail_sizes)) if nb_tail_clusters > 0  else 'NA'
    linkage_data = {NON_ISOLATED_PAGES: non_isolated_urls,
                    HEAD_PRODUCT: nb_head_clusters, TAIL_PRODUCTS: nb_tail_clusters,
                    AVG_HEAD_PRODUCT_SIZE: avg_head_size, AVG_TAIL_PRODUCT_SIZE: avg_tail_size,
                    MEDIAN_HEAD_PRODUCT_SIZE: median_head_size, MEDIAN_TAIL_PRODUCT_SIZE: median_tail_size,
                    'NB linkages': sum(len(urls) * (len(urls) - 1) / 2 for urls in pid2urls.values())
                    }
    return linkage_data

def attribute_analysis():
    """
    Outputs a CSV with list of attributes and characteristics associated
    :return:
    """
    sgen = adapter_factory.spec_factory().specifications_generator()
    ds = dataset.Dataset()

    for source in sgen:
        nb_pages = len(source.pages)
        att2size = collections.Counter()
        att2values2count = collections.defaultdict(bdsa_utils.counter_generator)
        att2url = collections.defaultdict(set)
        for url, specs in source.pages.items():
            att2size.update(specs.keys())
            for att, value in specs.items():
                att2values2count[att][value] += 1
                att2url[att].add(url)
        att2commonvalues = {att: values2count.most_common(4) for att, values2count in att2values2count.items()}

        #Now try to identify potential intra-source match
        value2att = collections.defaultdict(set)
        for att, values in att2commonvalues.items():
            for val in values:
                if val[0] not in ['yes','no','1','2','3']:
                    value2att[val[0]].add(att)
        att2similars = collections.defaultdict(set)
        for value, atts in value2att.items():
            for a1, a2 in itertools.combinations(atts,2):
                if len(att2url[a1] & att2url[a2]) == 0:
                    att2similars[a1].add(a2)
                    att2similars[a2].add(a1)
        for att, size in att2size.items():
            common_values = [x for x,y in att2commonvalues[att]]
            common_values += [''] * (4 - len(common_values))
            row = {'source': source.site, 'cat': source.category, 'name': att.replace(',', '-#-'), 'size': size,
                   'ratio': size/nb_pages, 'domain': len(att2values2count[att]),   'top1': common_values[0],
                   'top2': common_values[1], 'top3': common_values[2], 'top4': common_values[3],
                   'similars': str(att2similars[att])}
            ds.add_row(row)
    ds.export_to_csv(_config_.get_output_dir(), 'attribute_stats', False)

def detect_copies_in_dataset():
    """
    Detect pages and sources that are very similar, output in 2 csv files
    :return:
    """
    source_collection = adapter_factory.spec_factory().specifications_generator()
    ds_pages, ds_sources = _detect_copies_in_dataset_intern(source_collection)
    ds_sources.export_to_csv(_config_.get_output_dir(), 'copy_sources', True)
    ds_pages.export_to_csv(_config_.get_output_dir(), 'copy_pages',True)


def _detect_copies_in_dataset_intern(source_collection, min_similarity_pages=0.9, min_similarity_sources=.2, min_blocking_common_atts=3, max_size_group=8):
    """
    Cf detect_copies_in_dataset
    :param source_collection:
    :return:
    >>> sources = []
    >>> sources.append(tsp('s1').kv(1,100).kv(2,100).kv(3,100).p('u11')\
                       .kv(1,100).kv(2,100).kv(3,100).p('u12')\
                       .kv(5,100).kv(6,100).kv(7,100).p('u13')\
                       .kv(8,100).kv(9,100).kv(10,100).p('u14')\
                       .end())
    >>> sources.append(tsp('s2').kv(1,100).kv(2,100).kv(3,100).p('u21')\
                   .kv(1,100).kv(2,100).kv(3,100).kv(4, 50).p('u22')\
                   .kv(5,100).kv(6,100).kv(7,100).p('u23')\
                   .end())
    >>> res1, res2 = _detect_copies_in_dataset_intern(sources, min_similarity_pages=0.7)
    >>> res1.rows
    [{'source1': 's1__dummy', 'source2': 's1__dummy', 'url1': 'u11', 'url2': 'u12', 'title1': '', 'title2': '', 'spec1': frozenset(), 'spec2': frozenset()}, \
{'source1': 's1__dummy', 'source2': 's2__dummy', 'url1': 'u11', 'url2': 'u21', 'title1': '', 'title2': '', 'spec1': frozenset(), 'spec2': frozenset()}, \
{'source1': 's1__dummy', 'source2': 's2__dummy', 'url1': 'u11', 'url2': 'u22', 'title1': '', 'title2': '', 'spec1': frozenset(), 'spec2': frozenset({('4', '50')})}, \
{'source1': 's1__dummy', 'source2': 's2__dummy', 'url1': 'u12', 'url2': 'u21', 'title1': '', 'title2': '', 'spec1': frozenset(), 'spec2': frozenset()}, \
{'source1': 's1__dummy', 'source2': 's2__dummy', 'url1': 'u12', 'url2': 'u22', 'title1': '', 'title2': '', 'spec1': frozenset(), 'spec2': frozenset({('4', '50')})}, \
{'source1': 's1__dummy', 'source2': 's2__dummy', 'url1': 'u13', 'url2': 'u23', 'title1': '', 'title2': '', 'spec1': frozenset(), 'spec2': frozenset()}, \
{'source1': 's2__dummy', 'source2': 's2__dummy', 'url1': 'u21', 'url2': 'u22', 'title1': '', 'title2': '', 'spec1': frozenset(), 'spec2': frozenset({('4', '50')})}]

    >>> res2.rows
    [{'source1': 's1__dummy', 'source2': 's2__dummy', 'ratio_similarity': 0.75}, \
{'source1': 's2__dummy', 'source2': 's1__dummy', 'ratio_similarity': 1.0}, \
{'source1': 's1__dummy', 'source2': 'INTERNAL', 'ratio_similarity': 0.5}, \
{'source1': 's2__dummy', 'source2': 'INTERNAL', 'ratio_similarity': 0.6666666666666666}]

    """

    pair2elements = collections.defaultdict(set)
    url2specs = {}
    url2title = collections.defaultdict(str)
    source2url = {}
    for source in source_collection:
        source_metadata = source.metadata_only()
        source2url[source_metadata] = source.pages.keys()
        for url, specs in source.pages.items():
            spec_new = set()
            for key, value in specs.items():
                if key == '<page title>':
                    url2title[url] = value
                else:
                    value = string_utils.folding_using_regex(value)
                    pair2elements[(key, value)].add(url)
                    spec_new.add((key, value))
            url2specs[url] = frozenset(spec_new)
    mblocking = MetablockingGraph(min_blocking_common_atts)
    for elements in tqdm(pair2elements.values(), desc='Analyze each pair'):
        if len(elements) < max_size_group:
            mblocking.add_full_clique([{x: {1} for x in elements}], 1)
    ds_pages = dataset.Dataset([SOURCE_1, SOURCE2, URL_1, URL_2, TITLE, TITLE2, COMMON_SPEC, SPEC, SPEC2])
    ds_sources = dataset.Dataset([SOURCE_1, SOURCE2, RATIO_SIMILARITY])
    source2ext_source2url = collections.defaultdict(bdsa_utils.dd_set_generator)
    source2internal_url_copy = collections.defaultdict(set)
    url2source = bdsa_utils.build_dict(source2url.items(), lambda su: su[1], lambda su: su[0],
                                       multi=False, key_getter_returns_multi=True)
    for candidate in tqdm(sorted(mblocking.get_candidates().keys())):
        spec1 = url2specs[candidate[0]]
        spec2 = url2specs[candidate[1]]
        common_spec = spec1 & spec2
        if len(common_spec) / max(len(spec1), len(spec2)) > min_similarity_pages:
            source1 = url2source[candidate[0]]
            source2 = url2source[candidate[1]]
            if source1 != source2:
                source2ext_source2url[source1][source2].add(candidate[0])
                source2ext_source2url[source2][source1].add(candidate[1])
            else:
                source2internal_url_copy[source1].update({candidate[0], candidate[1]})
            ds_pages.add_row({SOURCE_1: str(source1), SOURCE2: str(source2), URL_1: candidate[0], URL_2: candidate[1],
                              TITLE: url2title[candidate[0]], TITLE2: url2title[candidate[1]],
                              # COMMON_SPEC: common_spec,
                              SPEC: spec1 - common_spec, SPEC2: spec2 - common_spec})
    for source1, esource2urls in source2ext_source2url.items():
        for source2, urls in esource2urls.items():
            rate_similarity = len(urls) / len(source2url[source1])
            if rate_similarity > min_similarity_sources:
                ds_sources.add_row({SOURCE_1: str(source1), SOURCE2: str(source2), RATIO_SIMILARITY: rate_similarity})
    for source, urls in source2internal_url_copy.items():
        containment = len(urls) / len(source2url[source])
        if containment > min_similarity_sources:
            ds_sources.add_row({SOURCE_1: str(source), SOURCE2: 'INTERNAL', RATIO_SIMILARITY: containment})
    return ds_pages, ds_sources

def detect_potential_linkages():
    bd = bdsa_data_transformed.BdsaDataTransformed(page2sa2value={})
    val2url = collections.defaultdict(set)
    url2title = {}
    drop = set()
    for source in adapter_factory.spec_factory().specifications_generator():
        for url, spec in source.pages.items():
            url2title[url] = url
            for key, value in spec.items():
                if key == '<page title>':
                    url2title[url] = value
                else:
                    val_transformed = bd.transform_value(None, value)
                    if val_transformed not in drop:
                        if val_transformed in val2url and len(val2url[val_transformed]) > 100:
                            del val2url[val_transformed]
                            drop.add(val_transformed)
                        else:
                            val2url[val_transformed].add(url)
    url2values = bdsa_utils.build_dict(val2url.items(), key_getter=lambda v2u:
                ["%s___%s" %(url2title[p[0]], url2title[p[1]]) for p in itertools.combinations(v2u[1],2)],
                          value_getter=lambda v2u:(v2u[0], len(v2u[1])), multi=True, key_getter_returns_multi=True)
    u2v_json = {}
    for url, val2sizes in url2values.items():
        if sum(1/v2s[1] for v2s in val2sizes) > 0.1:
            u2v_json[url] = ['%s__occ:%d' % ('-'.join(val_size[0]), val_size[1]) for val_size in val2sizes]
    io_utils.output_json_file(u2v_json, 'url2vals')


RATIO_SIMILARITY = 'ratio_similarity'
SPEC2 = 'spec2'
SPEC = 'spec1'
COMMON_SPEC = 'common_spec'
TITLE2 = 'title2'
TITLE = 'title1'
URL_2 = 'url2'
URL_1 = 'url1'
SOURCE2 = 'source2'
SOURCE_1 = 'source1'