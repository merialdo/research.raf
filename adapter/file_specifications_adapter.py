import json
import os

from tqdm import tqdm

from config.bdsa_config import _config_
from model.datamodel import SourceSpecifications
from utils import string_utils, io_utils
from adapter import abstract_specifications_adapter


SPEC_KEY = 'spec'
URL_KEY = 'url'


class FileSpecificationsGenerator(abstract_specifications_adapter.AbstractSpecificationsGenerator):

    def __init__(self, dataset_name=None, category=None):
        if dataset_name:
            self.spec_dir = _config_.get_spec_path_from_dataset_name(dataset_name)
        else:
            self.spec_dir = _config_.get_specifications()

        self.category = category or _config_.get_category()

    def nb_of_specs(self):
        nb = 0
        for asite in os.listdir(self.spec_dir):
            site_path = os.path.join(self.spec_dir, asite)
            if os.path.isdir(site_path):
                nb += len(os.listdir(site_path))
        return nb

    def source_names_ordered_linkage_decreasing(self):
        sources_list = []
        file_with_list_sources = os.path.join(self.spec_dir, 'sources_by_linkage.txt')
        if os.path.exists(file_with_list_sources):
            for source_name in io_utils.import_generic_file_per_line(file_with_list_sources):
                source_name = source_name.replace('\n','')
                source_split = source_name.split('__')
                if source_split[1] == _config_.get_category():
                    sources_list.append(SourceSpecifications(source_split[0], source_split[1], None))
        else:
            for asite in os.listdir(self.spec_dir):
                asite_fullpath = os.path.join(self.spec_dir, asite)
                if os.path.isdir(asite_fullpath):
                    for asource in os.listdir(asite_fullpath):
                        if "_spec.json" in asource:
                            category = asource.replace("_spec.json", "")
                            if category == _config_.get_category():
                                sources_list.append(SourceSpecifications(asite, category, None))
        return sources_list

    def _specifications_generator_intern(self, normalize_data=True):
        """
        Generator that returns a source with all associated data at a time
        :param directory: 
        :return: 
        """
        all_sources = []

        for asite in os.listdir(self.spec_dir):
            site_path = os.path.join(self.spec_dir, asite)
            if os.path.isdir(site_path):
                all_sources.extend((asite, asource) for asource in os.listdir(site_path) if '_spec.json' in asource)

        for source in tqdm(all_sources, desc='Retrieve all source...'):
            with open(os.path.join(self.spec_dir, source[0], source[1])) as data_file:
                category = source[1].replace("_spec.json", "")
                if not self.category or self.category == category:
                    specifications = json.load(data_file)
                    specifications_adapted = abstract_specifications_adapter. \
                        build_specifications_object(specifications, normalize_data, normalize_data, normalize_data)
                    yield SourceSpecifications(source[0], category, specifications_adapted)

    def _source_specifications_intern(self, site, category):
        """
        Returns specifications of a provided source
        :param directory: 
        :param site: 
        :param category: 
        :return: 
        """
        source_file1 = os.path.join(self.spec_dir, site, category + "_spec.json")
        with open(source_file1) as data_file:
            source_data = json.load(data_file)
            specifications = abstract_specifications_adapter \
                .build_specifications_object(source_data, True, False, True)
            return SourceSpecifications(site, category, specifications)

    def persist_specifications(self, sites2category2page2att2value, source_linkage_retriever=None):
        """
        Take in input a deserialized specification list and persist it in file system
        :param sites2category2page2att2value: 
        :return: 
        """
        new_dir = os.path.join(_config_.get_output_dir(),
                               'spec_subset' + string_utils.timestamp_string_format())
        os.mkdir(new_dir)
        # delete folder content?
        for site in sites2category2page2att2value:
            categories_data = sites2category2page2att2value[site]
            dirname = os.path.join(new_dir, site)
            os.mkdir(dirname)
            for category in categories_data:
                pages_json_format = _convert_internal_format_to_json_format(categories_data[category])
                io_utils.output_json_file(pages_json_format, directory=dirname,
                                          filename='%s_spec' % category, timestamp=False)
                if source_linkage_retriever:
                    linkage_data = source_linkage_retriever(SourceSpecifications(site, category, None))
                    io_utils.output_json_file(linkage_data, directory=dirname,
                                              filename='%s_linkage' % category, timestamp=False)


    def persist_specifications_functional(self, output_tag, source_linkage_retriever, source_spec_retriever):
        """
        Persist specifications using external functions
        :param source_linkage_retriever: generator method for linkage data for each source
        :param source_spec_retriever: generator method for spec data for each source
        :return:
        """
        output_dir = io_utils._build_filename(output_tag, None)
        os.mkdir(output_dir)
        for source in source_spec_retriever():
            source_output_dir = os.path.join(output_dir, source.site)
            os.mkdir(source_output_dir)
            source_data = _convert_internal_format_to_json_format(source.pages)
            io_utils.output_json_file(source_data, directory=source_output_dir, filename='%s_spec' % source.category,
                                      timestamp=False)
            if source_linkage_retriever:
                linkage_data = source_linkage_retriever(source)
                io_utils.output_json_file(linkage_data, directory=source_output_dir,
                                          filename='%s_linkage' % source.category, timestamp=False)

def _convert_internal_format_to_json_format(source_data):
    pages_to_serialize = []
    for url, specs in source_data.items():
        pages_to_serialize.append({SPEC_KEY: specs, URL_KEY: url})
    return pages_to_serialize
