import collections
import itertools

import community
import networkx as nx
import numpy

import adapter.abstract_specifications_adapter
from adapter import abstract_specifications_adapter, global_file_linkage
from config.bdsa_config import _config_
from model import dataset
from scripts.scripts_constants import *
from utils import stats_utils
from utils import string_utils

AVG_SIZE_OF_CLUSTER = 'avg_size_of_cluster'

NUMBER_OF_CLUSTERS = 'number_of_clusters'

MAX_HOMOGENEITY_TO_CLUSTER = 0.5

MIN_NODES_TO_CLUSTER = 10

URLS = 'urls'

NUMBER_OF_NODES = 'number_of_nodes'


## The homogeneity of a source or a product is computed in this way:
## for each couple of pages related to that source/product, compute jaccard similarity of attribute names
## Then compute indicators (such as mean, average, entropy) of this measure.
## The bigger jaccard similarity mean the more homogeneous a source/product is.

def compute_product_homogeneity():
    """
    For each product ID, compute homogeneity of that product
    See above for the meaning of homogeneity
    :return: 
    """
    product_measures = dataset.Dataset([NAME])
    url2stats = _build_url2stats()
    print("end building url2stats")
    idlinkage = global_file_linkage.GlobalLinkageFile()
    for id,cat2urls in idlinkage.get_full_map().items():
        all_urls = []
        for cat, urls in cat2urls.items():
            all_urls.extend(urls)
        specs = [url2stats[url] for url in all_urls if url in url2stats]
        if len(specs) >= 2:
            graph = _build_graph(specs)
            stats = _stats_on_graph(id,  cat, graph)
            product_measures.add_row(stats)
    stats_utils.compute_head_tail_dataset(product_measures, NUMBER_OF_NODES)
    product_measures.export_to_csv(_config_.get_output_dir(), 'product_homogeneity', True)

def compute_sources_homogeneity(do_clustering, export_csv):
    """
    For each source, compute homogeneity of that source
    See above for the meaning of homogeneity
    :return: 
    """
    source_measures = dataset.Dataset([NAME])
    spec_gen = abstract_specifications_adapter.specifications_generator(False, False)
    if do_clustering:
        sites2category2page2att2value = collections.defaultdict(dict)
    for source in spec_gen:
        print("Building graph for source %s %s" % (source.site, source.category))
        source_graph = _build_graph(source.pages)
        stats = _stats_on_graph(source.site, source.category, source_graph)
        stats[NUMBER_OF_PAGES] = len(source.pages)
        print("Stats OK for source %s %s" % (source.site, source.category))
        if do_clustering:
            _build_clustered_source(sites2category2page2att2value, source, source_graph, stats)

        source_measures.add_row(stats)
    stats_utils.compute_head_tail_dataset(source_measures, NUMBER_OF_PAGES)
    if export_csv:
        source_measures.export_to_csv(_config_.get_output_dir(), 'source_homogeneity', True)
    if do_clustering:
        adapter.abstract_specifications_adapter.persist_specifications(sites2category2page2att2value)

def _build_clustered_source(sites2category2page2att2value, source, source_graph, stats):
    """
    Put source in sites2category2page2att2value variable, clustering it if it is big and heterogeneous.
    :param sites2category2page2att2value: 
    :param source: 
    :param source_graph: 
    :param stats: 
    :return: some information about clustering
    """
    if stats[NUMBER_OF_NODES] >= MIN_NODES_TO_CLUSTER and stats[AVERAGE] < MAX_HOMOGENEITY_TO_CLUSTER:
        url2page = {page['url']: page for page in source.pages}
        communities = community.best_partition(source_graph)
        page_clusters = collections.defaultdict(list)
        for attribute_set, key in communities.items():
            all_urls = source_graph.node[attribute_set][URLS]
            page_clusters[key].extend(url2page[url] for url in all_urls)
        for key, pages in page_clusters.items():
            sites2category2page2att2value[source.site][source.category + "--" + str(key)] = pages
        nb_clusters = len(list(page_clusters.keys()))
    else:
        sites2category2page2att2value[source.site][source.category] = source.pages
        nb_clusters = 1
    stats[NUMBER_OF_CLUSTERS] = nb_clusters
    stats[AVG_SIZE_OF_CLUSTER] = stats[NUMBER_OF_PAGES] / float(nb_clusters)

### INTERN METHODS ###

def _build_url2stats():
    """
    Build a map with url--> specs for each page.
    This is useful for product analysis, as we have a map product --> list of page URLs
    :return: 
    """
    url2stats = {}
    for source in abstract_specifications_adapter.specifications_generator(True, False):
        for page in source.pages:
            url2stats[page['url']] = page
    return url2stats

def _build_graph(source_data):
    """
    Build a graph for source with nodes= page, edges=jaccard similarity between pages  
    :param source_data: 
    :return: 
    """
    source_graph = nx.Graph()

    #first, we build nodes grouping up all pages with same list of attribute names
    for url, specs in source_data.items():
        all_keys = frozenset(string_utils.folding_using_regex(key) for key in list(specs.keys()))
        source_graph.add_node(all_keys)
        node = source_graph.node[all_keys]
        if URLS not in node:
            node[URLS] = set()
        node[URLS].add(url)
    for couple in itertools.combinations(source_graph.nodes(), 2):
        distance = _attributes_name_jaccard_similarity(couple[0], couple[1])
        source_graph.add_edge(couple[0], couple[1], weight=distance)
    return source_graph

def _stats_on_graph(site, category, graph):
    """
    Computes some stats on edge weights of the graph, and put it in a dictionary
    :param element_name: 
    :param graph: 
    :return: 
    """
    result = {}
    result[NAME] = site + '__' + category
    result['__site'] = site
    result['__category'] = category
    all_edges = [edge[2]['weight'] for edge in graph.edges(data=True)]
    result[NUMBER_OF_NODES] = len(graph.nodes())
    result[AVERAGE] = numpy.mean(all_edges) if len(all_edges) > 0 else 1
    # result[STANDARD_DEV] = numpy.std(all_edges)
    # result[ENTROPY] = stats_utils.compute_entropy(all_edges, 0.1)
    return result

def _attributes_name_jaccard_similarity(page_spec1, page_spec2):
    """
    Returns jaccard distance between normalized attribute names 
    1-union/intersection
    :param page_spec1: 
    :param page_spec2: 
    :return: 
    """
    return float(len(page_spec1 & page_spec2)) / len(page_spec1 | page_spec2)

### TEST AND SCRIPTS

def _test():
    source1 = []
    source1.append({'url':'page_with_ab', 'spec': {'a':5, 'b':10}})
    source1.append({'url': 'page_with_ac', 'spec': {'a': 5, 'b': 10}})
    source1.append({'url': 'page_with_xy', 'spec': {'a': 5, 'c': 10}})
    G = _build_graph(source1)
    print(G.nodes())
    print(G.edges(data=True))
    stats = _stats_on_graph('lol', 'lo', G)
    ds = dataset.Dataset([NAME])
    ds.add_row(stats)
    ds.export_to_csv(_config_.get_output_dir(), 'test_homogeneity', True)

if __name__ == '__main__':
    # test()
    compute_sources_homogeneity(True, False)
    # compute_product_homogeneity()
    #cluster_datas()
