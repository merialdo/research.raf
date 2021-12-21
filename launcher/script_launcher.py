from scripts import json2sql, filter_dexter_dataset, \
    linkage_stats, source_product_homogeneity, source_stats, debug_alignment
from sys import argv
import time

### SQL methods ###

def sql_find_all_dbs():
    json2sql.execute_methods([json2sql.select_all_dbs])

def sql_insert_all_data(specifications, linkage, truncate_specifications, truncate_linkage):
    json2sql.insert_or_clear_data(specifications, linkage, truncate_specifications, truncate_linkage)

def sql_insert_test_data():
    json2sql.execute_methods([json2sql.insert_test_data])

### FILTER AND TRANSFORM SPECIFICATIONS DATASET ###

# These scripts make some transformations on the dataset, in order to remove some bad data and
# improve homogeneity of sources

def ftd_get_clean_dataset():
    filter_dexter_dataset.get_clean_dataset()

def ftd_3333_filtering():
    filter_dexter_dataset.build_filtered_subset()

def ftd_filter_linkage_components():
    linkage_stats.id2sources_graph(True, True)

def ftd_split_specifications_in_clusters():
    source_product_homogeneity.compute_sources_homogeneity(True, False)

def merge_cat_comm_linkage():
    filter_dexter_dataset.merge_cat_comm_linkage()

## After filtering dataset, filter record linkage file keeping only URLs that are in the dataset
## useful after cleaning tha dataset AND/OR after building a test subset of dataset
## ALSO keep only IDs associated to 2 pages (at least) and put min/max on character size

def ftd_filter_record_linkage():
    filter_dexter_dataset.filter_record_linkage_dexter_file(2, 1, 50)

# This method is useful to build a test set of specifications, along with filter_record_linkage
def ftd_build_random_subset():
    filter_dexter_dataset.build_random_subset(category='camera')


### MAKING STATS AND ANALYSIS OVER DATASET ###

#Compute homogeneity of sources or product, i.e. average ratio of common attribute names
# between pages of same source/product. See file for details.
# Sources with a good homogeneity are easier to align as the schema is more stable

def compute_source_homogeneity():
    ##here we do not cluster sources
    source_product_homogeneity.compute_sources_homogeneity(False, True)

def compute_product_homogeneity():
    source_product_homogeneity.compute_product_homogeneity()

# Part of schema used by each page on average (e.g. 5 pages each with 5 attributes,
# 10 distinct attributes globally --> average usage = 0.5).
# Also computes some other stats (% of common attributes,...), see class for details
def source_schema_analysis():
    source_stats.source_schema_analysis()

# General measures on a source
def compute_general_stats():
    source_stats.compute_general_stats(True)

# Compute cardinality and selectivity of attributes of sources
def compute_cardinality_selectivity():
    source_stats.compute_cardinality_selectivity()

# Stats on linkage conflicts
def compute_stats_on_conflicting_urls():
    linkage_stats.compute_stats_on_conflicting_urls()

# Computes components in source2id graph, outputs some statistics but does not filter the dataset
def compute_linkage_components():
    linkage_stats.id2sources_graph(Fa*lse, True)

# Build colormap with ID size (string) + nb of urls (per category)
def compute_length_id_string_and_number_of_urls():
    linkage_stats.compute_length_id_string_and_number_of_urls()

def print_linkage_graph():
    linkage_stats.print_linkage_graph('linkage_graph')

### EXPERIMENTS ###

# Build the GOLDEN SET
def build_golden_set():
    debug_alignment.build_golden_set()

# Build the GOLDEN SET
def compute_evaluation():
    debug_alignment.compute_evaluation()

## DETAILED ANALYSIS

# Find couple of sources with many linkages, for each category
def find_sources_with_many_linkages():
    linkage_stats.find_sources_with_many_linkages(output_nb_sources=0)

def compute_record_linkage_conflicts():
    linkage_stats.check_incoherence_in_record_linkage()

## TEST method (in case of any deployment problem)
def test_method():
    print('yololo')

# TODO compare sources

actions = {
    'sql_dbs': sql_find_all_dbs,
    'test': test_method,
    'connected_sources': find_sources_with_many_linkages,
    'id_size2nb_urls': compute_length_id_string_and_number_of_urls,
    'random_subset': ftd_build_random_subset,
    'filter_record_linkage': ftd_filter_record_linkage,
    'filter_linkage_components': ftd_filter_linkage_components,
    'compute_linkage_components': compute_linkage_components,
    'conflicting_urls': compute_stats_on_conflicting_urls,
    'build_golden_set': build_golden_set,
    'compute_evaluation': compute_evaluation,
    'build_clean_dataset': ftd_get_clean_dataset,
    'build_3333_filtered_dataset': ftd_3333_filtering,
    'cardinality_selectivity': compute_cardinality_selectivity,
    'print_linkage_graph': print_linkage_graph,
    'merge_cat_comm_linkage': merge_cat_comm_linkage,
    'general_stats': compute_general_stats,
    'source_schema_analysis': source_schema_analysis
} #TODO add all other scripts

# cf. https://stackoverflow.com/questions/46734798/write-python-code-that-execute-python-scripts

if __name__ == '__main__':
    method = argv[1]
    print('Running method %s'%(method))
    now = time.time()
    actions[method]()
    end = time.time()
    print('finished, took %f seconds'%(end-now))
