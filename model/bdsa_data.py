
import collections
from typing import Callable

from tqdm import tqdm

from adapter import adapter_factory
from config.bdsa_config import _config_
from model import datamodel, simulation
from model.bdsa_data_transformed import BdsaDataTransformed
from model.datamodel import SourceSpecifications, SourceAttribute, Page
from utils import bdsa_utils
from utils.bdsa_utils import list_padder
from config.bdsa_config import _config_

MAX_JACCARD_SIMILARITY_ATTRIBUTES = 0.8

class BdsaData:
    """
    Contains data imported from all sources of a category, with also some maps for efficient navigation and analysis
    """
    def __init__(self, simulation_input: simulation.Simulation, value_modifier):
        # Input data
        self.page2sa2value = collections.defaultdict(dict)
        self.source2pages = collections.defaultdict(set)
        self.simulation = simulation_input
        self.value_modifier = value_modifier
        self.excluded_attribute_names = _config_.get_excluded_attribute_names()

        self.linkage_adapter = adapter_factory.linkage_factory(_config_.get_linkage_suffix())


        # Those 2 are to build the domain of attributes
        self.sa2value2occs = collections.defaultdict(bdsa_utils.counter_generator)
        self.sa2size = collections.Counter()

        # In which pages an attribute occurs. If 2 attributes occur in same page, they are not equivalent
        self.sa2urls = collections.defaultdict(set)

        # Most frequent values (non-normalized) in each source
        self.sa2topvalues = {}

        # Initial clustering of attributes and pages
        self.pid2source2pages = collections.defaultdict(bdsa_utils.dd_set_generator)

        # Generated attributes
        self.source2generated_sas = collections.defaultdict(set)
        # Keeps inverted index of IDS, maybe useful for some specific analysis (remove inital linkages
        # from list of linkages to evaluate
        if _config_.debug_mode():
            self.url2pid = {}

        self._transformed_data = None

    def _reset_data(self):
        """
        This method should be called in any moment in which data are modified, so any generated information is removed
        :return:
        """
        self._transformed_data = None

    def get_transformed_data(self):
        """
        Get all transformed data
        :return:
        """
        if not self._transformed_data:
            self._transformed_data = BdsaDataTransformed(self.page2sa2value)
        return self._transformed_data

    def add_source(self, source_complete: SourceSpecifications):
        """
        Add a single source to data
        :param source_complete:
        :param pids_getter:
        :param value_modifier:
        :return:
        """
        self._reset_data()
        source_metadata = source_complete.metadata_only()
        # We keep a list of values non-normalized for debugging output
        att2values_non_normalized = collections.defaultdict(bdsa_utils.counter_generator)
        for url, key2values in source_complete.pages.items():
            pids = self.linkage_adapter.ids_by_url(url, source_complete.site, source_complete.category)
            page = datamodel.page_factory(url, source_metadata)
            self.simulation.check_and_add_product_page(self, page, key2values, pids)
            self._update_att2value(att2values_non_normalized, key2values, source_metadata)
        # TODO: how to manage this in case of newly created attributes?
        for sa, values in att2values_non_normalized.items():
            res = bdsa_utils.most_common_deterministic(values, 5)
            list_padder(res)
            self.sa2topvalues[sa] = res

    def _update_att2value(self, att2values_non_normalized, key2values, source_metadata):
        """
        Increase list att2values_non_normalized, useful for debugging
        :param att2values_non_normalized:
        :param excluded:
        :param key2values:
        :param source_metadata:
        :return:
        """
        for key, value in key2values.items():
            if key not in self.excluded_attribute_names:
                sa = datamodel.source_attribute_factory(source_metadata.category, source_metadata.site, key)
                att2values_non_normalized[sa][value] += 1

    def end_algorithm(self):
        """
        Should be called when algorithm has ended: all pages/other elements that where removed for training should be now
        added for evaluation.
        :return:
        """
        self.simulation.do_delayed_operations(self)

    def finalize_import(self):
        self.simulation.potentially_remove_linkage_clusters(self.pid2source2pages)
        self.simulation.add_wrong_linkages(self)

    def add_page(self, page:Page, key2values:dict, pids):
        """
        Add a specific page to data
        :param source_metadata: source with site and category
        :param url: url of this page
        :param key2values: specifications in page
        :param pids: Product IDs of page
        :return:
        """
        self.source2pages[page.source].add(page)
        if _config_.debug_mode() and len(pids) > 0:
            self.url2pid[page.url] = pids
        for pid in pids:
            self.pid2source2pages[pid][page.source].add(page)
        for key, value in key2values.items():
            if key not in self.excluded_attribute_names:
                norm_value = self.value_modifier(value)
                self._intern_add_occurrence(key, norm_value, page)

    def _intern_add_occurrence(self, att_name, norm_value, page, att_suffix=None):
        """
        Add occurrences, internal method used both for standard source importing and for generated attributes
        :param att_name:
        :param norm_value:
        :param page:
        :return:
        """
        sa = datamodel.source_attribute_factory(page.source.category, page.source.site, att_name, att_suffix)
        if sa not in self.page2sa2value[page]:
            self.sa2value2occs[sa][norm_value] += 1
            self.sa2size[sa] += 1
        else:
            old_value = self.page2sa2value[page][sa]
            print ("*** WARNING: duplicate entry for sa %s for page url %s [old value %s, new %s ]" % (sa.name, page.url, old_value, norm_value))
            if self.sa2value2occs[sa][old_value] == 1:
                del self.sa2value2occs[sa][old_value]
            else:
                self.sa2value2occs[sa][old_value] -= 1
        self.page2sa2value[page][sa] = norm_value
        self.sa2urls[sa].add(page.url)
        return sa


    def finalize_generated(self):
        """
        To be called after last generated attribute occurrence has been added.
        Builds the top values map
        :return:
        """
        for source, sas in self.source2generated_sas.items():
            for sa in sas:
                #Find top 5 elements
                res = sorted(self.sa2value2occs[sa].items(), key=lambda t: t[1], reverse=True)[:5]
                self.sa2topvalues[sa] = list_padder(res)

    def remove_all_generated(self):
        """
        Removes all generated attributes
        :return:
        """
        if len(self.source2generated_sas) > 0:
            self._reset_data()
            for source, sas in tqdm(dict(self.source2generated_sas).items(), desc='Removing generated...'):
                for sa in list(sas):
                    self.remove_attribute(sa)
            self.source2generated_sas.clear()

    def remove_attribute(self, sa:SourceAttribute):
        """
        Remove a provided attribute from all list and maps
        :param remove_from_generated_list: if True, remove element from generated list also.
        :param sa:
        :return:
        """
        self._reset_data()
        urls = self.sa2urls[sa]
        for url in urls:
            page = datamodel.page_factory(url, sa.source)
            self.page2sa2value[page].pop(sa, None)
        if sa in self.sa2urls:
            del self.sa2urls[sa]
        if sa in self.sa2value2occs:
            del self.sa2value2occs[sa]
        if sa in self.sa2size:
            del self.sa2size[sa]
        if sa in self.sa2topvalues:
            del self.sa2topvalues[sa]

        if sa in self.source2generated_sas[sa.source]:
            self.source2generated_sas[sa.source].remove(sa)


    def nb_attributes(self):
        return len(self.sa2size)

    def nb_generated_atts(self):
        return sum(len(sas) for sas in self.source2generated_sas.values())

    def nb_original_attributes(self):
        return self.nb_attributes() - self.nb_generated_atts()

    GeneratedAttributeOccurrence = collections.namedtuple("GeneratedAttributeOccurrence", ['name', 'value','page','suffix'])

    def launch_attribute_generation(self, attribute_generator, custom_filter=lambda sa, bdsa:True):
        """
        Add generated attributes at once, filtering the most useful ones.
        :param attribute_generator: a generator method that yields all attribute occurrences to generate
        :param custom_filter: potential additional filter for generated attributes. Return TRUE if attribute is ok
        :return:

        """
        self.remove_all_generated()
        #Here we check for potential duplicated attributes
        sa2page2value = collections.defaultdict(dict)
        for new_att in attribute_generator:
            sa = self._intern_add_occurrence(new_att.name, new_att.value, new_att.page, new_att.suffix)
            sa2page2value[sa][new_att.page] = new_att.value
            self.source2generated_sas[new_att.page.source].add(sa)
        self._remove_duplicates(sa2page2value)
        self._reset_data()
        transformed = self.get_transformed_data() # Keep it for comparison
        for sa in tqdm(sa2page2value.keys(), desc='Filtering generated atts...'):
            if self._virtual_attribute_should_be_removed(custom_filter, sa, transformed):
                self.remove_attribute(sa)

        self.finalize_generated()

    def _remove_duplicates(self, sa2page2value):
        """
        Remove duplicated attributes (ie attributes who always appear in same pages, with same values and extracted from the same original attribute)
        :param sa2page2value:
        :return:
        """
        sa2page_value = {sa: tuple(sorted((page, value) for page, value in page2value.items())) for sa, page2value in
                         sa2page2value.items()}
        orig_name_page_value_seen = set()
        for sa, page_value in sorted(sa2page_value.items()):
            orig_name_page_value = (sa.get_original_name(), page_value)
            if orig_name_page_value in orig_name_page_value_seen:
                self.remove_attribute(sa)
                del sa2page2value[sa]
            else:
                orig_name_page_value_seen.add(orig_name_page_value)

    def _virtual_attribute_should_be_removed(self, custom_filter, sa, transformed):
        # min 10% of attribute values and 3 values
        min_ratio_values_vs_original_attribute = self.sa2size[sa] / \
                                                 self.sa2size[
                                                     sa.get_original_attribute()] > _config_.get_min_value_ratio_on_original_attribute()
        min_absolute_nb_values = self.sa2size[sa] >= _config_.get_min_values()
        # for each distinct value, sum of 1/[nb_clusters_value_occurs].
        do_remove = not min_ratio_values_vs_original_attribute or not min_absolute_nb_values \
                    or not custom_filter(sa, self) \
                    or _too_similar(transformed.get_transformed_value2occs(sa),
                                    transformed.get_transformed_value2occs(sa.get_original_attribute()))
        return do_remove


def _too_similar(virtual_domain: collections.Counter, original_domain: collections.Counter):
    """
    Verify if domain of generated and original attributes are too similar (jaccard similarity)
    :param virtual_domain:
    :param original_domain:
    :return:
    """
    intersection = virtual_domain & original_domain
    union = virtual_domain | original_domain
    return sum(intersection.values()) / sum(union.values()) > MAX_JACCARD_SIMILARITY_ATTRIBUTES
