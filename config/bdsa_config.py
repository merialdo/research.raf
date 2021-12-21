import collections
from configparser import ConfigParser, ExtendedInterpolation
import os
from enum import Enum

import project_constants
from model import datamodel
from model.datamodel import SourceAttribute

dir = os.path.dirname(__file__)
_candidates_ = [os.path.join(dir, x) for x in ['defaults.ini', 'local_config.ini', 'algo_parameters.ini']]

class BdsaConfig:
    """
    Adapter for python SafeConfigParser configuration.
    Provides specific methods for some configuration options.
    
    DOCUMENTATION CAN BE FOUND ON defaults.ini config file. This is just a bridge 
    """
    def __init__(self, candidates):
        self.candidates = candidates
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config_to_override = collections.defaultdict(dict)
        self.reset_config()
        # Specific config that can be set INSIDE the program, that is specified overwrites any other config.

    def reset_config(self):
        self.config_to_override['inputs']['root_dir'] = project_constants.ROOT_DIR
        found = self.config.read(self.candidates)
        missing = set(self.candidates) - set(found)
        print('Found config files:', sorted(found))
        print('Missing config files     :', sorted(missing))
        self.master_attributes = MasterAttribute(self)
        self._overwrite_config()

    def activate_debug(self):
        """
        !!! RESET MUST BE CALLED AFTER ALL MODIFICATIONS!!!
        :return:
        """
        self.config_to_override['output']['debug_mode'] = 'yes'
        self._overwrite_config()

    def _overwrite_config(self):
        """
        Specific config that can be set INSIDE the program, that is specified overwrites any other config.
        This method should be called after each modification in self.config_to_override AND after reset.
        :return:
        """
        self.config.read_dict(self.config_to_override)

    def get_option(self, section, option):
        return self.config.get(section, option)

    def get_list(self, section, option):
        elements = self.config.get(section, option)
        return elements.split(',') if len(elements) > 0 else []

    def get_yes_no(self, section, option):
        return self.config.get(section, option) == 'yes'

    def get_file_relative_to_input(self, filepath):
        input_dir = self.config.get('local', 'input_path')
        return os.path.join(input_dir, filepath)

    ### Specific dataset

    def get_specifications(self):
        return self.config.get('inputs', 'specifications_full_path')

    def get_ground_truth_path(self):
        return self.config.get('inputs', 'ground_truth_path')

    def get_ground_truth_instance_level_path(self):
        return self.config.get('inputs', 'ground_truth_instance_level_path')

    ### These methods can be used to specify dataset and ground truth indipendently from config ####

    def get_spec_path_from_dataset_name(self, name):
        return os.path.join(self.config.get('local', 'input_path'), 'dataset', name)

    def get_ground_truth_path_from_dataset_name(self, name):
        return os.path.join(self.config.get('local', 'input_path'), 'ground_truth', '%s.csv' % name)

    def get_ground_truth_instance_level_path_from_dataset_name(self, name):
        return os.path.join(self.config.get('local', 'input_path'), 'ground_truth', '%s_instance.csv' % name)

    ### END

    def get_category(self):
        return self._get_value_or_none('inputs', 'category')

    def _get_value_or_none(self, section, option):
        cat = self.config.get('%s' % section, option)
        if not (cat) or len(cat) == 0:
            return None
        return cat

    def get_output_dir(self):
        return self.config.get('local', 'output_path')

    def get_cache_dir(self):
        return self.config.get('local', 'cache_path')

    def get_experiments_output_dir(self):
        return self.config.get('local', 'experiments_output_path')

    def get_algo_parameters_repository_path(self):
        return self.config.get('local', 'algo_parameters_repository_path')

    def get_linkage_dexter(self):
        return self.config.get('inputs', 'linkage_dexter_full_path')

    def get_community_linkage_dexter(self):
        return self.config.get('inputs', 'community_linkage_dexter_full_path')

    def get_linkage_dexter_combined_clean(self):
        return self.config.get('inputs', 'linkage_dexter_combined_clean_full_path')

    def get_linkage_dexter_combined(self):
        return self.config.get('inputs', 'linkage_dexter_combined_full_path')

    def get_linkage_suffix(self):
        return self.config.get('inputs', 'linkage_suffix')

    ## ALGORITHM PARAMETERS ##

    # Input

    def get_excluded_attribute_names(self):
        return self.get_list('inputs', 'excluded_attribute_names')

    # Representation parameters

    def get_clusters_to_compare(self):
        clusters = self.get_list('output', 'clusters_to_compare')
        res = [[int(x) for x in cluster_group.split('-')] for cluster_group in clusters]
        return res

    def do_output_main_analysis(self):
        return self.get_yes_no('output', 'do_output_main_analysis')

    def do_output_ikgpp(self):
        return self.IkgppMode[self.config.get('output', 'ikgpp_mode')] != self.IkgppMode.NO

    def do_output_ikgpp_original_attributes(self):
        return self.IkgppMode[self.config.get('output', 'ikgpp_mode')] == self.IkgppMode.ORIGINAL



    class IkgppMode(Enum):
        NO = 1  # No IKGPP in output
        ORIGINAL = 2  # Original attribute values will be shown in provenances
        TAGGED = 3  #  Tagged values extracted in virtual attribute values will be shown

    def do_synthetic_evaluation(self):
        return self.get_yes_no('output', 'do_synthetic_evaluation')

    def do_separate_isolated(self):
        return self.get_yes_no('output', 'do_separate_isolated')

    def get_specific_nodes(self, category):
        results = []
        if self.config.has_option('output', 'specific_nodes_'+category):
            list = self.get_list('output', 'specific_nodes_'+category)
            for element in list:
                elem_array = element.split('/')
                results.append(datamodel.source_attribute_factory(category, elem_array[0], elem_array[1]))
        return results

    ## Edge filters
    def get_max_frequency_single_value(self):
        return self.config.getfloat('algorithm', 'max_frequency_single_value')

    ## Other algorithm parameters

    def get_default_linkage_weight(self):
        return self.config.getfloat('algorithm', 'default_linkage_weight')
    def get_rate_matches_threshold(self):
        return self.config.getfloat('parameters', 'rate_matches_threshold')
    def get_error_rate_per_value(self):
        return self.config.getfloat('algorithm', 'error_rate_per_value')
    def get_default_apriori_equivalence_ratio(self):
        return self.config.getfloat('algorithm', 'default_apriori_equivalence_ratio')
    def get_default_apriori_linkage_ratio(self):
        return self.config.getfloat('algorithm', 'default_apriori_linkage_ratio')
    def get_similarity_names_weight(self):
        return self.config.getfloat('algorithm', 'similarity_names_weight')
    def get_min_edge_weight(self):
        return self.config.getfloat('algorithm', 'min_edge_weight')
    def get_min_edge_weight_linkage(self):
        return self.config.getfloat('algorithm', 'min_edge_weight_linkage')
    def do_allow_linkage_same_source(self):
        return self.get_yes_no('algorithm', 'allow_linkage_same_source')
    def do_exclude_generated_from_family_linkage(self):
        return self.get_yes_no('algorithm', 'exclude_generated_from_family_linkage')


    class RecordLinkageMethod(Enum):
        NONE = 1  # No linkage is computed, only 1 iteration of schema alignment
        PROB = 2  # Probabilistic linkage method based on matches on aligned attributes
        CLASS = 3  # Classifier based on equality of aligned attributes

    def do_use_original_linkage_for_training_in_iteration(self):
        return self.get_yes_no('algorithm', 'use_original_linkage_for_training_in_iteration')

    def get_record_linkage_method(self) -> RecordLinkageMethod:
        return self.RecordLinkageMethod[self.config.get('algorithm', 'record_linkage_method')]

    def do_restart_linkage(self):
        return self.get_yes_no('algorithm', 'restart_linkage')

    def do_keep_previous_original_clusters(self):
        return self.get_yes_no('algorithm', 'keep_previous_original_clusters')

    class RecordLinkageBehavior(Enum):
        ADD = 1  # Just add linkage and never delete it.
        DELETE = 2  # Just delete linkage, and never add new ones.
        BOTH = 3  # Do both

    def get_record_linkage_behavior(self) -> RecordLinkageBehavior:
        return self.RecordLinkageBehavior[self.config.get('algorithm', 'record_linkage_behavior')]

    def switch_record_linkage_behavior(self):
        current = self.RecordLinkageBehavior[self.config.get('algorithm', 'record_linkage_behavior')]
        if current == self.RecordLinkageBehavior.ADD:
            _next = self.RecordLinkageBehavior.DELETE
        elif current == self.RecordLinkageBehavior.DELETE:
            _next = self.RecordLinkageBehavior.ADD
        else:
            _next = self.RecordLinkageBehavior.BOTH
        self.config.set('algorithm', 'record_linkage_behavior', _next.name)

    def get_neg_sample_linkage(self):
        return self.config.getint('algorithm', 'neg_sample_linkage')

    def get_min_blocking_score_linkage(self):
        return self.config.getfloat('algorithm', 'min_blocking_score_linkage')

    def debug_mode(self):
        return self.get_yes_no('output', 'debug_mode')

    def get_number_of_iterations(self):
        return self.config.getint('algorithm', 'number_of_iterations')

    def use_idf(self):
        return self.get_yes_no('algorithm', 'use_idf')

    def delete_subclusters_only_virtuals(self):
        return self.get_yes_no('algorithm', 'delete_subclusters_only_virtuals')

    OneValuePerProductLegacy = Enum('OneValuePerProductLegacy', 'yes no no_minweight')

    def one_value_per_product(self):
        return self.OneValuePerProductLegacy[self.config.get('algorithm','one_value_per_product')]

    def exclude_same_page_for_generated(self):
        return self.get_yes_no('algorithm', 'exclude_same_page_for_generated')

    def compute_posterior_probability(self):
        return self.get_yes_no('algorithm', 'compute_posterior_probability')

    def get_clustering_algorithm(self):
        return self.Clustering[self.config.get('algorithm', 'clustering_algorithm')]

    class Clustering(Enum):
        LOUVAIN = 1  # Louvain clustering algorithm (maximize modularity)
        CC = 2  # Connected components (weight-independent)
        AGGLO = 3  # Connected components, weigth > 0.5 (make configurable?), avoid same source attributes in same cluster,
        # by removing lowest weight edge

    ### TAGGING

    class ExtractDictValuesFrom(Enum):
        HEAD_CLUSTERS = 1 #  Extract dictionary values only from attributes in HEAD clusters
        NON_ISOLATED = 2 # Extract dictionary values from all non-isolated clusters

    class AttributesToTag(Enum):
        TAIL_CLUSTERS = 1 #  Tag only isolated attributes and attributes in tail clusters
        ALL_ATTRIBUTES = 2 # Tag ALL attributes in the dataset

    def get_extract_dict_values_from(self):
        return self.ExtractDictValuesFrom[self.config.get('tagging', 'extract_dict_values_from')]

    def get_attributes_to_tag(self):
        return self.AttributesToTag[self.config.get('tagging', 'attributes_to_tag')]

    def get_max_len_atomic_values(self):
        return self.config.getint('tagging', 'max_len_atomic_values')

    def get_min_len_atomic_values(self):
        return self.config.getint('tagging', 'min_len_atomic_values')

    def get_max_ratio_occs(self):
        return self.config.getfloat('tagging', 'max_ratio_occs')

    def do_assign_snippets_one_cluster(self):
        return self.get_yes_no('tagging', 'assign_snippets_one_cluster')

    def do_debug_tags(self):
        return self.get_yes_no('output', 'debug_tags')

    def get_min_att_in_cluster(self):
        return self.config.getint('tagging', 'min_att_in_cluster')

    def get_min_common_values_attributes(self):
        return self.config.getint('algorithm', 'min_common_values_attributes')

    def get_min_ngram(self):
        return self.config.getint('tagging', 'min_ngram')

    def get_max_ngram(self):
        return self.config.getint('tagging', 'max_ngram')

    def get_min_value_ratio_on_original_attribute(self):
        return self.config.getfloat('tagging', 'min_value_ratio_on_original_attribute')

    def get_min_values(self):
        return self.config.getint('tagging', 'min_values')

    def get_min_distinct_values_score(self):
        return self.config.getfloat('tagging', 'min_distinct_values_score')

    def add_extraction_data_in_output(self):
        return self.get_yes_no('tagging', 'add_extraction_data_in_output')

    def get_tag_every_combination(self):
        return self.get_yes_no('tagging', 'tag_every_combination')

    def get_min_mi(self):
        return self.config.getfloat('tagging', 'min_mi')

    def get_min_significant_tokens(self):
        return self.config.getint('tagging', 'min_significant_tokens')

    def get_min_distinct_values_generated_mi(self):
        return self.config.getint('tagging', 'min_distinct_values_generated_mi')

    def get_min_sa_pair(self):
        return self.config.getint('tagging', 'min_sa_pair')

    def get_min_page_pair(self):
        return self.config.getint('tagging', 'min_page_pair')

    def get_min_clustering_similarity_to_stop_iteration(self):
        return self.config.getfloat('algorithm', 'min_clustering_similarity_to_stop_iteration')

    def get_min_elements_in_identical_clusters_to_stop_iteration(self):
        return self.config.getfloat('algorithm', 'min_elements_in_identical_clusters_to_stop_iteration')

    def get_min_dict_similarity_to_stop_iteration(self):
        return self.config.getfloat('algorithm', 'min_dict_similarity_to_stop_iteration')

    def get_min_terms_in_identical_entries_to_stop_iteration(self):
        return self.config.getfloat('algorithm', 'min_terms_in_identical_entries_to_stop_iteration')

    ## INPUTS AND ADAPTERS ##

    def get_specifications_source(self):
        """
        Source of specifications and linkage data
        :return: 
        """
        return self.SpecificationsSource[self.config.get('inputs', 'specifications_source')]

    class SpecificationsSource(Enum):
        FILE = 1 # Find specifications on file system
        MONGO = 2 # Load specifications from mongo

    def get_mongo_host(self):
        return self.config.get('local','mongo_host')

    def get_mongo_db(self):
        return self.config.get('local', 'mongo_db')

    ##### MASTER SOURCES #####
    def get_master_source(self):
        return self._get_value_or_none('master_source', 'source_name')

    def get_master_attributes(self):
        return self.get_list('master_source', 'source_attributes')

    def get_excluded_master_attributes(self):
        return self.get_list('master_source', 'excluded_source_attributes')

    def get_min_size_external_clusters(self):
        return self.config.getint('master_source', 'min_size_external_clusters')

    def get_common_token_ratio(self):
        return self.config.getfloat('algorithm', 'common_token_ratio')

    ### SIMULATION ###

    def do_delete_pages_without_linkage(self):
        return self.get_yes_no('simulation','delete_pages_without_linkage')

    def get_percentage_linkage_kept(self):
         return self.config.getfloat('simulation', 'percentage_linkage_kept')

    def get_linkage_removal_rule(self):
         return LinkageRemovalRule[self.config.get('simulation', 'linkage_removal_rule')]

    def get_ratio_linkage_added(self):
        return self.config.getfloat('simulation', 'ratio_linkage_added')

    def get_number_sources_kept(self):
        return self.config.getint('simulation', 'number_sources_keep')

    def get_source_removal_rule(self):
        return SourceRemovalRule[self.config.get('simulation', 'source_removal_rule')]

    def get_sources_mandatory(self):
        return self.get_list('simulation', 'sources_mandatory')

    ### EXPERIMENTS ###
    def do_experiments_attribute_cardinality(self):
        return self.get_yes_no('experiments', 'attribute_cardinality')

    def do_experiments_attribute_size(self):
        return self.get_yes_no('experiments', 'attribute_size')

    def do_experiments_cluster_distinct_sources(self):
        return self.get_yes_no('experiments', 'cluster_distinct_sources')

    def do_experiments_attribute_linkage(self):
        return self.get_yes_no('experiments', 'attribute_linkage')

    def do_experiments_source_size(self):
        return self.get_yes_no('experiments', 'source_size')

    def do_experiments_source_linkage(self):
        return self.get_yes_no('experiments', 'source_linkage')

class SourceRemovalRule(Enum):
    RANDOM = 1
    KEEP_MORE_LINKAGE = 2  # Keep sources with most linkage first
    KEEP_LESS_LINKAGE = 3  # Keep sources with less linkage
    NONE = 4 # No removal of sources
    #linkage_removal_rule

class LinkageRemovalRule(Enum):
    RANDOM = 1
    KEEP_SMALL_CLUSTERS = 2  # Keep smallest clusters (delete big cluster first)
    KEEP_BIG_CLUSTERS = 3  # Keep biggest clusters (delete small cluster first)
    NONE = 4  # No removal of linkage
    # linkage_removal_rule

class MasterAttribute:
    """
    Checks if an attribute is a master attribute
    """
    def __init__(self, config: BdsaConfig):
        self.master_source = config.get_master_source()
        self.master_attributes = config.get_master_attributes()
        self.no_master_attributes = len(self.master_attributes) == 0
        self.master_excluded_atts = config.get_excluded_master_attributes()
        self.no_master_excluded_attributes = len(self.master_excluded_atts) == 0

    def is_master_attribute(self, sa:SourceAttribute):
        return self.master_source and sa.source.site == self.master_source and \
               (self.no_master_attributes or sa.get_original_name() in self.master_attributes) and \
               (self.no_master_excluded_attributes or sa.get_original_name() not in self.master_excluded_atts)


_config_ = BdsaConfig(_candidates_)

if __name__ == '__main__':
    print(_config_.get_option('inputs', 'specifications'))
    print(_config_.get_specifications())