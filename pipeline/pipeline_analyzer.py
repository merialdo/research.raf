import bisect
import collections

from cachetools import LRUCache, cached
from tqdm import tqdm

from adapter import synthetic_dataset_adapter
from config import constants
from config.bdsa_config import _config_
from config.constants import CLUSTER_ID, OCCURRENCES, SOURCE_NAME, CARDINALITY, CLUSTER_SIZE, TOP_3, TOP_2, TOP_1, \
    LINKED_PAGES
from model import dataset, datamodel
from model.bdsa_data import BdsaData
from pipeline.pipeline_abstract import AbstractPipeline
from pipeline.pipeline_common import analyze_clustering_results, ClusteringOutput
from utils import string_utils, stats_utils, io_utils, experiment_utils

FULL_NAME = 'full_name'

ISOLATED_ANALYSIS = 'Isolated Analysis'

GLOBAL = 'Global'

SA_MEASURES = 'Schema alignment measures'
PAGE_MEASURES = 'Page linkage measures'

F_MEASURE = 'F1'
RECALL = 'R'
PRECISION = 'P'

"""
Compute some statistics on schema alignment results, and outputs it in a readable way
"""

INTRA_EDGE = 'INTRA_EDGE'
EXTRA_EDGE = 'EXTRA_EDGE'
NODE = 'NODE'

TYPE = 'type'

WEIGHT = 'weight'

FILENAME_GRAPH_BUILT_CACHE = "bdsa_output"

class PipelineAnalyzer(AbstractPipeline):
    """
    Build algorithm output, some stats and analysis. Also builds the instance-level output.
    """
    def __init__(self, tag_input, cluster_occurrences_isolated_atts=False):
        """

        :param tag_input:
        :param cluster_occurrences_isolated_atts: if true, at the instance-level cluster together all occs of isolated
        attributes (ie considers them as atomic by default).
        """
        self._tag = tag_input
        self._cluster_occurrences_isolated_atts = cluster_occurrences_isolated_atts

    def run(self, data: tuple):
        output = data[0]
        debug_stats = data[1]
        for cat, cat_data in output.items():
            stats = self.compute_stats_on_graph(cat_data)
            if cat in debug_stats:
                stats.update(debug_stats[cat])
            if _config_.do_output_main_analysis():
                ds_synthesis, ds_details, isolated = self.get_clustering_results(cat_data)
                ds_matches = self._build_matches_csv(cat_data)
                if _config_.do_separate_isolated():
                    isolated.export_to_csv(_config_.get_output_dir(), "%s_%s_%s" % ('isolated', self._tag, cat), True)
                else:
                    ds_details.rows.extend(isolated.rows)
                ds_synthesis.export_to_csv(_config_.get_output_dir(), "%s_%s_%s" % ('cluster_synthesis', self._tag, cat),
                                           True)
                ds_details.export_to_csv(_config_.get_output_dir(), "%s_%s_%s" % ('cluster_detail', self._tag, cat),
                                         True)

                if _config_.debug_mode():
                    ds_matches.export_to_csv(_config_.get_output_dir(),
                                             "%s_%s_%s" % ('cluster_matches', self._tag, cat), True)
                    data_family_linkage = dataset.Dataset(
                        ['source1', 'source2', 'att1', 'att2', 'val1', 'val2', 'info', 'url1', 'url2','iteration'])
                    for rows in cat_data.page_matches:
                        data_family_linkage.add_rows(rows)
                    data_family_linkage.export_to_csv(_config_.get_output_dir(), "%s_%s_%s" % ('family_linkage_differences', self._tag, cat),
                                             True)

            io_utils.output_json_file(stats, "%s_%s_%s" % (FILENAME_GRAPH_BUILT_CACHE, self._tag, cat))
            instance_level = self._build_instance_level_clusters(cat_data)
            io_utils.output_json_file(instance_level, "%s_%s_%s" % ('instance_level', self._tag, cat))
            if _config_.do_output_ikgpp():
                ikgpp = self._build_ikgpp(cat_data)
                io_utils.output_json_file(ikgpp, "%s_%s_%s" % ('ikgpp', self._tag, cat))

    def name(self):
        return "Analyzer"

    def need_input(self):
        return True

    def need_output(self):
        return False

    def compute_stats_on_graph(self, clustering_output: ClusteringOutput):
        stats = analyze_clustering_results(clustering_output)
        if _config_.do_synthetic_evaluation():
            self.compute_evaluation(stats, clustering_output)

        #isolated nodes are excluded from count
        att_cluster_sizes = [sum(len(sas) for sas in source2sas.values()) for source2sas in
                               clustering_output.sa_clusters.values()]
        pages_cluster_sizes = [sum(len(pages) for pages in source2pages.values()) for source2pages in
                               clustering_output.page_clusters.values()]

        # if _config_.debug_mode():
        #     stats['Linkage differencies'] = self.analyze_linkage_changes(clustering_output)
        stats[constants.STATS_ATTR_CLUSTER_SIZES] = stats_utils.GroupSize(att_cluster_sizes).__dict__
        stats[constants.STATS_PAGE_CLUSTER_SIZES] = stats_utils.GroupSize(pages_cluster_sizes).__dict__
        return stats

    def analyze_linkage_changes(self, clustering_output: ClusteringOutput):
        """
        Analyze how the linkage changed after iterative linkage-alignment
        :return: 
        """
        old_linkage = collections.defaultdict(set)
        new_linkage_clusters = []
        old_isolated = []
        for source2pages in clustering_output.page_clusters.values():
            pages_pid_flattened = [page for pages in source2pages.values() for page in pages]
            new_linkage_clusters.append(pages_pid_flattened)
            for page in pages_pid_flattened:
                pids = clustering_output.bdsa_data.url2pid.get(page.url, [])
                for pid in pids:
                    old_linkage[pid].add(page)
                if len(pids) == 0:
                    #add to isolated cluster
                    old_isolated.append([page])
        for page in clustering_output.page_isolated:
            new_linkage_clusters.append([page])
            #TODO duplicated code
            pids = clustering_output.bdsa_data.url2pid.get(page.url, [])
            for pid in pids:
                old_linkage[pid].add(page)
            if len(pids) == 0:
                # add to isolated cluster
                old_isolated.append([page])
        old_linkage_clusters = list(old_linkage.values())
        old_linkage_clusters.extend(old_isolated)
        new_link, old_link, union_link = experiment_utils.evaluate_expected_computed({}, new_linkage_clusters, old_linkage_clusters)[3:6]
        return {'TOTAL LINKAGES NOW':new_link, 'TOTAL old linkages': old_link, 'Linkages kept': union_link,
                'New linkages created': new_link - union_link, 'Linkages deleted': old_link - union_link}

    def compute_evaluation(self, stats, clustering_output: ClusteringOutput):
        ## TODO those lines are specific to the type of input (synth/real)
        expected_sa_clusters, computed_sa_clusters = synthetic_dataset_adapter.golden_set_for_synthetic_data(
            clustering_output.sa_clusters, clustering_output.sa_isolated,
            lambda sa: sa.attname, lambda sa: synthetic_dataset_adapter.SyntheticSourceAttribute(sa))

        expected_page_clusters, computed_page_clusters = synthetic_dataset_adapter.golden_set_for_synthetic_data(
            clustering_output.page_clusters, clustering_output.page_isolated,
            lambda page: page.pid, lambda page: synthetic_dataset_adapter.SyntheticPage(page))

        source_categories = {'H_Source': lambda x: x.synth_source.ht == 'H', 'T_Source': lambda x: x.synth_source.ht == 'T',
                      'Missing linkage > 0.5': lambda x: x.synth_source.linkage_missing > 0.5,
                      'Missing linkage < 0.5': lambda x: x.synth_source.linkage_missing < 0.5,
                      'Linkage error 0.1': lambda x: x.synth_source.linkage_error == 0.1,
                      'Linkage error 0.01': lambda x: x.synth_source.linkage_error == 0.01,
                      'Value error 0.1': lambda x: x.synth_source.value_error == 0.1,
                      'Value error 0.01': lambda x: x.synth_source.value_error == 0.01
                      }

        sa_categories = dict(source_categories)
        sa_categories.update({
            'H_Attributes': lambda x: x.att_ht == 'H', 'T_Attributes': lambda x: x.att_ht == 'T',
            'Card2': lambda x: x.cardinality == 2, 'Card3': lambda x: x.cardinality == 3,
            'Card7': lambda x: x.cardinality == 7, 'Card10': lambda x: x.cardinality == 10,
            'Card10+': lambda x: x.cardinality > 10
        })
        global_sa_results, specific_sa_results = experiment_utils.evaluate_dataset(computed_sa_clusters, expected_sa_clusters.values(),
                                                                                   sa_categories)
        global_page_results, specific_page_results = experiment_utils.evaluate_dataset(computed_page_clusters, expected_page_clusters.values(),
                                                                             source_categories)
        stats[SA_MEASURES] = {}
        stats[SA_MEASURES][GLOBAL] = {PRECISION: global_sa_results.precision, RECALL: global_sa_results.recall,
                                      F_MEASURE: global_sa_results.f_measure}
        for cat, results in specific_sa_results.items():
            stats[SA_MEASURES][cat] = {PRECISION: results.precision, RECALL: results.recall,
                                       F_MEASURE: results.f_measure}

        stats[PAGE_MEASURES] = {}
        stats[PAGE_MEASURES][GLOBAL] = {PRECISION: global_page_results.precision, RECALL: global_page_results.recall,
                                      F_MEASURE: global_page_results.f_measure}
        for cat, results in specific_page_results.items():
            stats[PAGE_MEASURES][cat] = {PRECISION: results.precision, RECALL: results.recall,
                                       F_MEASURE: results.f_measure}

    def get_clustering_results(self, clustering_output: ClusteringOutput):
        """
        Outputs 2 csv files representing the algorithm output
        * "cluster synthesis": a list of clusters, each with ID, number of attributes and most frequent attribute names
        * "cluster details": a list of attributes, each with cluster ID it belongs and most frequent attribute values
        :return:
        """

        synhtesis = dataset.Dataset([constants.CLUSTER_ID, OCCURRENCES, TOP_1, TOP_2, TOP_3])
        details = dataset.Dataset([constants.CLUSTER_ID, CLUSTER_SIZE, SOURCE_NAME, FULL_NAME, TOP_1, TOP_2, TOP_3,
                                   CARDINALITY, OCCURRENCES, constants.NAME])

        isolated = dataset.Dataset([SOURCE_NAME, constants.NAME, CARDINALITY, OCCURRENCES, TOP_1, TOP_2, TOP_3])

        isolated_id = 0
        sa2linkage = self._build_sa2linkedpages(clustering_output.bdsa_data)

        for sa in clustering_output.sa_isolated:
            row = self._sa_to_detail_row(clustering_output, sa, sa2linkage)
            row.update({CLUSTER_ID: 'ISOLATED_%d' % isolated_id, CLUSTER_SIZE: 1})
            isolated_id += 1
            isolated.add_row(row)

        for cid, source2sas in tqdm(clustering_output.sa_clusters.items(), desc='Build cluster output'):
            cluster_sas = [sa for sas in source2sas.values() for sa in sas]
            # for each cluster, find most common attribute names (but try to remove too similar names, see sort_attnames for details)
            attname2occs = stats_utils.count_elements(cluster_sas, lambda x: string_utils.folding_using_regex(x.name))
            att_name_sorted = self._sort_attnames(attname2occs)

            row = {CLUSTER_ID: cid, OCCURRENCES: len(cluster_sas), TOP_1: att_name_sorted[0], TOP_2: att_name_sorted[1],
                   TOP_3: att_name_sorted[2]}
            synhtesis.add_row(row)
            for cluster_sa in cluster_sas:
                row = self._sa_to_detail_row(clustering_output, cluster_sa, sa2linkage)
                row.update({CLUSTER_ID: cid, CLUSTER_SIZE: len(cluster_sas)})
                details.add_row(row)
        for sa in tqdm(clustering_output.sa_deleted, desc='Add deleted attributes...'):
            row = self._sa_to_detail_row(clustering_output, sa, sa2linkage)
            details.add_row(row)
        return synhtesis, details, isolated

    def _build_sa2linkedpages(self, bdsa_data:BdsaData):
        already_analyzed_pages = set()
        sa2linked_pages = collections.Counter()
        for source2pages in bdsa_data.pid2source2pages.values():
            for pages in source2pages.values():
                for page in pages - already_analyzed_pages:
                    sa2linked_pages.update(bdsa_data.page2sa2value[page].keys())
        return sa2linked_pages


    def _sa_to_detail_row(self, clustering_output, elem, sa2linkage:collections.Counter):
        """
        Convert a source attribute into a row 
        :param clustering_output: 
        :param elem: 
        :return: 
        """
        domain = clustering_output.bdsa_data.sa2value2occs[elem]
        # now add rows to cluster details CSV.
        # For each attribute, find the 3 most frequent attribute values
        words = clustering_output.bdsa_data.sa2topvalues.get(elem, ['NA', 'NA', 'NA'])
        transformer = clustering_output.bdsa_data.get_transformed_data()
        words_with_transformations = [
            (val, #transformer.transform_value(elem, string_utils.folding_using_regex(val)),
             occs) for val, occs in words]

        row = {SOURCE_NAME: elem.source.site, FULL_NAME: elem.name, constants.NAME: elem.get_original_name(),
               CARDINALITY: len(domain), OCCURRENCES: sum(domain.values()), TOP_1: words_with_transformations[0],
               TOP_2: words_with_transformations[1], TOP_3: words_with_transformations[2], LINKED_PAGES: sa2linkage[elem]}
        return row

    def _sort_attnames(self, attname2occs, max_number_of_att_names=3):
        """
        Return the top 3 attribute names (most used in a cluster)
        Eliminate att names too similar to former ones (if we can find at least 3)
        Example: most frequent attribut names, sorted desc:
            --> max product size, size, product, max, viewfinder type, viewfinder, type
            --> OUTPUT:  max product size, max, viewfinder type
        :param attname2occs: 
        :param max_number_of_att_names: 
        :return: 
        """
        att_name_sorted = sorted(attname2occs.keys(), key=lambda elem: attname2occs[elem], reverse=True)
        att_name_list = []
        att_name_removed = []
        i = 0
        while i < len(att_name_sorted) and len(att_name_list) < max_number_of_att_names:
            if any(att_name_sorted[i].split(' ') <= elem.split(' ') for elem in att_name_list):
                att_name_removed.append(att_name_sorted[i])
            else:
                att_name_list.append(att_name_sorted[i])
            i += 1
        att_name_list.extend(att_name_removed)
        return att_name_list[0:3] + [''] * (max_number_of_att_names - len(att_name_list))

    def _build_matches_csv(self, cat_data: ClusteringOutput):
        """
        Builds a dataset with all matches between attributes
        :param cat_data:
        :return:
        """
        headers = ['source1', 'full name1', 'source2', 'full name2', 'score', 'prior']
        for i in range(1,4):
            headers.extend(['val_X%d' % i, 'val_Y%d' % i, 'nb%d' % i])
        match = dataset.Dataset(headers)
        for sa1, sas in cat_data.att_matches.items():
            for sa2, data in sas.items():
                # TODO add classes to make it more readable
                row = {'full name1': sa1.name, 'full name2': sa2.name,
                       'source1': sa1.source.site, 'source2': sa2.source.site,
                        'score': data['score'], 'prior': data['prior']}
                matches = data['matches']
                for i, el in enumerate(matches):
                    sign = 1 if el[0][0] == el[0][1] else -1
                    row['val_X%d' % (i+1)] = el[0][0]
                    row['val_Y%d' % (i+1)] = el[0][1]
                    row['nb%d' % (i+1)] = el[1] * sign
                match.add_row(row)
        return match

    AttributeInstanceCluster = collections.namedtuple('AttributeInstanceCluster',
                                                  'dictionary atomic nonatomic')

    def _build_instance_level_clusters(self, clustering_output:ClusteringOutput):
        bdsa_data = clustering_output.bdsa_data
        cid2names = clustering_output.find_name_for_clusters()
        output = {}
        for cid, source2sas in clustering_output.sa_clusters.items():
            cluster_name = self.__build_clustertag(cid, cid2names[cid])
            cluster_dict = set()
            atomic_attributes = set()
            non_atomic_attributes = set()
            for source, sas in source2sas.items():
                for sa in sas:
                    if not(sa.is_generated()):
                        cluster_dict.update(bdsa_data.sa2value2occs[sa].keys())
                        atomic_attributes.add(str(sa))
                    else:
                        non_atomic_attributes.add(str(sa.get_original_attribute()))
            output[cluster_name] = self.AttributeInstanceCluster(
                sorted(cluster_dict), sorted(atomic_attributes), sorted(non_atomic_attributes)
            )._asdict()
        return output


    def _build_ikgpp(self, clustering_output: ClusteringOutput):
        """
        Build integrated graph from original data
        :param clustering_output:
        :return:
        """
        source2aids = self._build_source2aids(clustering_output)

        pid2aid2provenances = dict()
        ### !!!! Note we do not use computed linkage but initial one
        bdsa_data = clustering_output.bdsa_data

        cid2names = clustering_output.find_name_for_clusters()
        # Iterate over entities (attribute IDs)
        for pid, source2pages in bdsa_data.pid2source2pages.items():
            aid2provenances = collections.defaultdict(list)
            aid2prov = collections.defaultdict(set) # To deduplicate
            for source, pages in source2pages.items():

                # For each source, we build a map sa --> AID so we can then build a AID 2 provenance dict
                sa2aid_involved = {sa: aid for aid in source2aids[source] for sa in clustering_output.sa_clusters[aid][source]}
                for page in pages:
                    page_sa2values = bdsa_data.page2sa2value[page]
                    #sas_in_page = sa2aid_involved.keys() & page_sa2values.keys()
                    for sa, value in page_sa2values.items():
                        # Ignore deleted SA (currently SA isolated from which a non-isolated v.a. was created)
                        if sa in clustering_output.sa_deleted:
                            continue
                        # !! This is an important aspect
                        # If an attribute is isolated then, according to _cluster_occurrences_isolated_atts, either we
                        # consider it as atomic (clustering all of its occs), either we do not provide in ikgpp, consider
                        # all its occs as isolated
                        if sa in sa2aid_involved:
                            sa_aid = sa2aid_involved.get(sa, None)
                            cluster_tag = self.__build_clustertag(sa_aid, cid2names[sa_aid])
                        else:
                            if self._cluster_occurrences_isolated_atts:
                                cluster_tag = 'ISOLATED_%s' % str(sa)
                            else:
                                continue
                        sa_to_show = sa.get_original_attribute() if _config_.do_output_ikgpp_original_attributes() else sa
                        provenance = datamodel.Provenance(page.url, sa_to_show, page_sa2values[sa_to_show])
                        if provenance not in aid2prov[cluster_tag]:
                            bisect.insort(aid2provenances[cluster_tag], str(provenance))
                            aid2prov[cluster_tag].add(provenance)
            pid2aid2provenances[pid] = aid2provenances

        return pid2aid2provenances

    @cached(cache=LRUCache(maxsize=1024))
    def __build_clustertag(self, cluster_id:int, cluster_name:str):
        return '%s__%d' % (cluster_name, cluster_id)

    def _build_source2aids(self, clustering_output):
        # Build inverted dict source2aid
        source2aid = collections.defaultdict(set)
        for aid, source2sas in clustering_output.sa_clusters.items():
            for source in source2sas.keys():
                source2aid[source].add(aid)
        return source2aid