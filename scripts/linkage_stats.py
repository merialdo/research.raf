import collections
import json

import itertools
import networkx as nx

from networkx.readwrite import json_graph
from tqdm import tqdm

from adapter import abstract_specifications_adapter, global_file_linkage, adapter_factory
from config import constants
from config.bdsa_config import _config_
from model import dataset, datamodel
from scripts.scripts_constants import *
from utils import stats_utils, string_utils, io_utils

### STATISTICS ABOUT RECORD LINKAGE
RATIO_HEAD_SOURCES = 'ratio head sources'
RATIO_CONFLICTING_URLS = 'Ratio conflicting URLs'
RATIO_MULTIPLE_URLS = 'Ratio multiple URLs'
RATIO_URLS_IN_LINKAGE = 'Ratio URLs in linkage'

NUMBER_OF_COMPONENTS = 'number_of_components'
HEAD_TAIL = '__HEAD-TAIL'
COMPONENT_INDEX = '__component_index'
NUMBER_OF_HEAD_SOURCES = 'number_head_sources'
SOURCE = 'source'
ALL_SOURCES = 'source_all'

CATEGORY = '__cat'
SITE = '__site'

NUMBER_OF_SOURCES = 'number_sources'

TYPE = 'type'
##no enum otherwise json serialization become complicated
ID = 'id'
SOURCE = 'source'

NUMBER_OF_CONFLICTING_URLS = 'Number_of_conflicting_URLs'
NUMBER_OF_URLS_IN_LINKAGE = 'Number_of_URLs_in_linkage'
NUMBER_OF_URLS_WITH_MULTIPLE_IDS = 'Number_of_URLs_affected_to_multiple_IDs'

def check_incoherence_in_record_linkage():
    """
    Verify if URLS that normalizes in the same way (eg: www.ebay.com/... and ebay.com/....
    have the same list of associated IDs.
    If not, outputs the IDs
    :return: 
    """

    duplicated_ids = set()
    duplicated_urls = set()

    with open(_config_.get_linkage_dexter(), 'r') as infile:
        normurl2urls2ids = collections.defaultdict(lambda: collections.defaultdict(list))
        cache = json.load(infile)
        for id,c2u in cache.items():
            for cat, urls in c2u.items():
                for url in urls:
                    url_norm = string_utils.url_normalizer(url)
                    normurl2urls2ids[url_norm][url].append(id)
        for norm_url, url2ids in normurl2urls2ids.items():
            all_ids = list(url2ids.values())
            if not(all(x==all_ids[0] for x in all_ids)):
                print('WRONG elements: %s'%(url2ids))
                for ids in all_ids:
                    duplicated_ids.update(ids)
                duplicated_urls.update(list(url2ids.keys()))

    print('ids: %s, urls: %s'%(str(len(duplicated_ids)), str(len(duplicated_urls))))

def id2sources_graph(output_specification_subset_components, output_synthesis_dataset):
    """
    Builds a bipartite graph ids <----> sources for each category
    Checks number of connected components, size and diameter
    :param output_specification_subset_components: if true, build a subset of specifications with only sources 
    in head linkage  components(i.e. with common products with many other sources)
    :param output_synthesis_dataset: if true, outputs a CSV file with informations on linkage components
    :return: 
    """

    # cat2graphs = io_utils.get_or_create_cache_file('linkage_graph', "%s_OO_%s"%
    #                                                (_config_.get_specifications(), _config_.get_linkage()),
    #                                                _build_linkage_graphs)
    cat2graphs = _build_linkage_graphs()

    ## for each node add attribute telling if source is head or tail according to #pages
    for element in list(cat2graphs.values()):
        stats_utils.compute_head_tail([n_d for n_d in element.nodes(True) if n_d[1][TYPE] == SOURCE],
                                      lambda node: node[1][NUMBER_OF_PAGES],
                                      lambda node, ht: stats_utils.assign_value(node[1], stats_utils.HT, ht))


    print('start computing components')
    #now compute connected components for each category, with some stats

    output = {}
    dataset_analysis = collections.defaultdict(lambda: dataset.Dataset([NAME]))

    for cat, graph in cat2graphs.items():
        components = nx.connected_component_subgraphs(graph)
        nb_comp = 0
        for comp in components:
            print('computing component in category %s ...'%(cat))
            all_nodes = comp.nodes(data=True)
            nb_ids = sum(1 for n,d in all_nodes if d[TYPE] == ID)
            all_sources_nodes = [(n,d) for n,d in all_nodes if d[TYPE] == SOURCE]
            nb_sources = len(all_sources_nodes)
            nd_head_sources = sum(1 for n, d in all_sources_nodes if d[stats_utils.HT] == stats_utils.HEAD)
            nb_pages = sum(d[NUMBER_OF_PAGES] for (n,d) in all_sources_nodes)
            component_name = cat + '__' + str(nb_comp)
            dataset_analysis[cat].add_row({NAME: component_name, CATEGORY: cat, COMPONENT_INDEX: nb_comp,
                                           NUMBER_OF_URLS_IN_LINKAGE: nb_ids, NUMBER_OF_SOURCES: nb_sources,
                                           NUMBER_OF_HEAD_SOURCES: nd_head_sources, NUMBER_OF_PAGES: nb_pages,
                                           ALL_SOURCES: [n for n,d in all_sources_nodes]})
            nb_comp += 1

        ## now group connected components in head/tail according to number of sources
        # (note that this is different from head/tail sources)
        stats_utils.compute_head_tail_dataset(dataset_analysis[cat], NUMBER_OF_SOURCES)

    if output_specification_subset_components:
        _build_specification_subset_components(dataset_analysis)

    if output_synthesis_dataset:
        _build_synthesis_dataset(cat2graphs, dataset_analysis)


def _build_synthesis_dataset(cat2graphs, dataset_analysis):
    """
    Generates a CSV file with informations on linkage of the sources 
    :param cat2graphs: 
    :param dataset_analysis: 
    :return: 
    """
    synthesis_dataset = dataset.Dataset([NAME])
    del cat2graphs
    for cat in list(dataset_analysis.keys()):
        for head_or_tail in [stats_utils.HEAD, stats_utils.TAIL]:
            elements = [x for x in dataset_analysis[cat].rows if x[stats_utils.HT] == head_or_tail]
            nb_sources = sum(x[NUMBER_OF_SOURCES] for x in elements)
            nb_head_sources = sum(x[NUMBER_OF_HEAD_SOURCES] for x in elements)
            nb_ids = sum(x[NUMBER_OF_URLS_IN_LINKAGE] for x in elements)
            nb_pages = sum(x[NUMBER_OF_PAGES] for x in elements)
            label = cat + '__' + head_or_tail
            synthesis_dataset.add_row({NAME: label, CATEGORY: cat,
                                       HEAD_TAIL: head_or_tail, NUMBER_OF_SOURCES: nb_sources,
                                       NUMBER_OF_HEAD_SOURCES: nb_head_sources,
                                       NUMBER_OF_URLS_IN_LINKAGE: nb_ids, NUMBER_OF_PAGES: nb_pages,
                                       NUMBER_OF_COMPONENTS: len(elements)})
    synthesis_dataset.export_to_csv(_config_.get_output_dir(), 'linkage_components', True)

def find_sources_with_many_linkages(nb_sources = None, only_head=False,
                                    base_site=None, subset:set=None, output_nb_sources=30):
    """
    Finds couple of sources with lots of linkages.
    I.e.: the top source is the one with most associated IDs
    Then, each subsequent source is the one with most common ID with the union of IDs of all
    sources found until now.
    :return: 
    """
    cat2graphs = _build_linkage_graphs()

    cat2data = {}
    for cat, graph in cat2graphs.items():
        print ("Category %s..." % cat)
        if only_head:
            stats_utils.compute_head_tail([n_d1 for n_d1 in graph.nodes(True) if n_d1[1][TYPE] == SOURCE],
                                          lambda node: node[1][NUMBER_OF_PAGES],
                                          lambda node, ht: stats_utils.assign_value(node[1], stats_utils.HT, ht))
            sources_to_return = set(node[0] for node in graph.nodes(data=True)
                                                       if node[1][TYPE] == SOURCE and
                                                        node[1][stats_utils.HT] == stats_utils.HEAD)
        elif subset:
            #return only sources in subset AND remove IDs not linked to at least 2 sources of subset
            sources_to_return = set(node[0] for node in graph.nodes(data=True) if node[1][TYPE] == SOURCE\
                                 and node[0] in subset)
        else:
            sources_to_return = set(node[0] for node in graph.nodes(data=True) if node[1][TYPE] == SOURCE)
        for idnode in tqdm([node[0] for node in graph.nodes(data=True) if node[1][TYPE] == ID],
                           desc='Removing IDs not linked to at least 2 sources of subset'):
            sources_linked_to_id = set(graph.neighbors(idnode)) & sources_to_return
            if len(sources_linked_to_id) < 2:
                graph.remove_node(idnode)
        print("Pre-processing computed....")


        ## build list of couple (source, associated IDs)
        nb_sources_this_cat = nb_sources or (len(subset) if subset else len(graph.nodes()))
        source2ids = {node: set(graph.neighbors(node)) for node in sources_to_return}

        #first source: a specific one OR the one with most associated IDs
        if base_site:
            top1_source = base_site[cat]
            id_list = source2ids.get(top1_source)
        else:
            top1_source, id_list = max(iter(source2ids.items()), key=lambda x_y2: len(x_y2[1]))

        top_sources = [(top1_source, len(id_list))]
        union_all_ids = set(id_list)
        nb_ids = len(union_all_ids)
        del source2ids[top1_source]

        #TODO rimettere tqdm o simili?
        pbar = tqdm(total=nb_sources_this_cat, desc='Analyzing sources...')
        while len(top_sources) < nb_sources_this_cat and nb_ids != 0 and len(source2ids) >= 1:
            nb_ids = _compute_next_top_page(source2ids, top_sources, union_all_ids)
            pbar.update()
        pbar.close()

        cat2data[cat] = top_sources

    if output_nb_sources > 0:
        specs = adapter_factory.spec_factory()
        output_sources = collections.defaultdict(set)

    for cat, datas in cat2data.items():
        if output_nb_sources and output_nb_sources > 0:
            for data in datas[:output_nb_sources]:
                output_sources[data[0].site].add(data[0].category)
        print ("*** %s ***\n" % cat)
        for data in datas:
            print ("%s\t%s"%(data[0].site, data[1]))
        for source in set(sources_to_return) - set(d[0] for d in datas):
            print("%s\t%s" % (source.site, 0))
        print ('\n\n')
    if output_nb_sources > 0:
        sites2category2page2att2value = {}
        for site, cats in output_sources.items():
            cat2pages = {cat: specs.source_specifications(site, cat).pages for cat in cats}
            sites2category2page2att2value[site] = cat2pages
        specs.persist_specifications(sites2category2page2att2value)


def _compute_next_top_page(source2ids, top_sources, union_all_ids):
    source_with_common_ids = [(source, len(ids & union_all_ids)) for source, ids in source2ids.items()]
    next_top_source, nb_ids = max(source_with_common_ids, key=lambda x_y: x_y[1])
    union_all_ids |= source2ids[next_top_source]
    top_sources.append((next_top_source, nb_ids))
    # remove the element retrieved
    del source2ids[next_top_source]
    return nb_ids

def print_linkage_graph(tag: str):
    cat2graphs = _build_linkage_graphs()
    for cat, graph in cat2graphs.items():
        print("Printing linkage graph for category %s..."%(cat))
        result = {}
        for node in graph.nodes(data=True):
            if node[1][TYPE] == SOURCE:
                result[str(node[0])] = [str(id) for id in graph[node[0]]]
        io_utils.output_json_file(result, "%s_%s_linkage_graph"%(tag, cat), timestamp=False)


def _build_linkage_graphs() -> dict:
    """
    Internal method to build a bipartite graph ids <--> sources PER category
    :return: 
    """
    cat2graphs = collections.defaultdict(lambda: nx.Graph())
    linkage = adapter_factory.linkage_factory()
    sgen = adapter_factory.spec_factory().specifications_generator()
    # build a graph with node=ids + sources, edge = the source contains at least 1 page with that id
    print("Build linkage graph...")
    for source in sgen:
        source_label = source.metadata_only()
        category = source.category
        cat2graphs[category].add_node(source_label, **{TYPE: SOURCE, NUMBER_OF_PAGES: len(source.pages)})
        associated_ids = set()
        for url in source.pages:
            page_ids = linkage.ids_by_url(url, source.site, source.category)
            associated_ids.update(page_ids)
        for pid in associated_ids:
            cat2graphs[category].add_node(pid, **{TYPE: ID})
            cat2graphs[category].add_edge(pid, source_label)
    del linkage
    return cat2graphs


def _build_specification_subset_components(dataset_analysis):
    sites2category2page2att2value = collections.defaultdict(lambda: collections.defaultdict(list))
    for dset in list(dataset_analysis.values()):
        for element in dset.rows:
            if element[stats_utils.HT] == stats_utils.HEAD:
                for source in element[ALL_SOURCES]:
                    spec_gen = adapter_factory.spec_factory()
                    sites2category2page2att2value[source.site][source.category] = spec_gen.\
                        source_specifications(source.site, source.category).pages
        spec_gen.persist_specifications(sites2category2page2att2value)


def compute_stats_on_conflicting_urls():
    """
    For each source, compute number of conflicting URLS (associated to ID present in different categories)
     and multiple URLs (associated to many ids)
    :return: 
    """
    idlinkage = global_file_linkage.GlobalLinkageFile()
    conflicting_urls = set([])
    url_in_linkage = set([])
    multi_id_urls = set([])
    for id,cat2urls in idlinkage.get_full_map().items():
        is_conflicting = len(cat2urls) > 1
        for cat, urls in cat2urls.items():
            if is_conflicting:
                conflicting_urls.update(urls)
            for url in urls:
                if url in url_in_linkage:
                    multi_id_urls.add(url)
                else:
                    url_in_linkage.add(url)

    source_dataset = dataset.Dataset([NAME])
    spec_gen = abstract_specifications_adapter.specifications_generator(False, False)
    for source in spec_gen:
        source_urls = set([page['url'] for page in source.pages])
        row = {NAME: str(source), SITE: source.site, CATEGORY: source.category, NUMBER_OF_PAGES: len(source.pages)}
        row[NUMBER_OF_CONFLICTING_URLS] = len(conflicting_urls & source_urls)
        row[NUMBER_OF_URLS_IN_LINKAGE] = len(url_in_linkage & source_urls)
        row[NUMBER_OF_URLS_WITH_MULTIPLE_IDS] = len(multi_id_urls & source_urls)
        row[RATIO_CONFLICTING_URLS] = row[NUMBER_OF_CONFLICTING_URLS] / float(len(source_urls))
        row[RATIO_MULTIPLE_URLS] = row[NUMBER_OF_URLS_WITH_MULTIPLE_IDS] / float(len(source_urls))
        row[RATIO_URLS_IN_LINKAGE] = row[NUMBER_OF_URLS_IN_LINKAGE] / float(len(source_urls))
        source_dataset.add_row(row)
    stats_utils.compute_head_tail_dataset(source_dataset, NUMBER_OF_PAGES)
    source_dataset.export_to_csv(_config_.get_output_dir(), 'conflicting_ids', True)

def compute_nb_linkages():
    """
    Compute total number of linkages in global file, in form id2category2urls (this is the dirty record linkage file)
    
    :return: 
    """
    with open(_config_.get_linkage_dexter(), 'r') as linkage_file:
        id2comm2urls = json.load(linkage_file)
        total_linkages = set()
        for comm2urls in id2comm2urls.values():
            for urls in comm2urls.values():
                urls.sort()
                total_linkages.update((a[0], a[1]) for a in itertools.combinations(urls, 2))
    print("Nb of output elements: %d" % len(total_linkages))