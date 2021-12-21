import itertools

import collections
from sys import argv

from adapter import output_and_gt_adapter, adapter_factory
from config.bdsa_config import _config_
from model import datamodel, dataset
from model.clusters import FreeClusterRules
from model.dataset import Dataset
from pipeline import cluster_utils
from pipeline.cluster_utils import WeightedEdge
from utils import bdsa_utils, string_utils

ID_REAL = 'cluster_id_real'

DebugSamename = collections.namedtuple('DebugSamename', 'asource aname bsource bname expected common1 common2 common3'
                                                        ' a1 a2 a3 b1 b2 b3')

WordBase = collections.namedtuple('WordBase','word folded')

def debug_samename(file_position):
    """
    Analyze attribute pairs with same name but different clusters in output
    :param file_position: 
    :return: 
    """

    ds = Dataset(DebugSamename._fields)
    category = _config_.get_category()
    cdo = output_and_gt_adapter.import_cluster_detail_csv('cluster_id', file_position, True, category)
    ground_truth = output_and_gt_adapter.import_cluster_detail_csv(ID_REAL, _config_.get_ground_truth_path(), category)

    sa2val2occs = collections.defaultdict(bdsa_utils.counter_generator)
    spec_factory = adapter_factory.spec_factory()
    for source in spec_factory.specifications_generator():
        for url, specs in source.pages.items():
            for key, value in specs.items():
                sa2val2occs[datamodel.source_attribute_factory(category, source.site, key)][value] += 1

    sas = sorted(cdo.sa2clusters.keys(), key=lambda sa: (sa.get_original_name(), sa.name, sa.source))
    for name, sas in itertools.groupby(sas, key=lambda  sa: sa.get_original_name()):
        for sa1, sa2 in itertools.combinations(sas, 2):
            # Attributes have same name but different output clusters
            # Note that here we do not detect instance-level part of attributes that should be merged
            if len(cdo.sa2clusters[sa1] & cdo.sa2clusters[sa2]) == 0 \
                    and sa1 in ground_truth.sa2clusters\
                    and sa2 in ground_truth.sa2clusters:
                same_cluster_expected = len(ground_truth.sa2clusters[sa1] & ground_truth.sa2clusters[sa2]) > 0
                common = collections.Counter({string_utils.folding_using_regex(k): v
                                            for k, v in sa2val2occs[sa1].items()}) & \
                         collections.Counter({string_utils.folding_using_regex(k): v
                                            for k, v in sa2val2occs[sa2].items()})

                only1 = collections.Counter({k:v for k,v in sa2val2occs[sa1].items()
                                             if string_utils.folding_using_regex(k) not in common.keys()})
                only2 = collections.Counter({k:v for k,v in sa2val2occs[sa2].items()
                                             if string_utils.folding_using_regex(k) not in common.keys()})
                top1 = bdsa_utils.list_padder(bdsa_utils.most_common_deterministic(only1, 3), 3, '')
                top2 = bdsa_utils.list_padder(bdsa_utils.most_common_deterministic(only2, 3), 3, '')
                common = bdsa_utils.list_padder(bdsa_utils.most_common_deterministic(common, 3), 3, '')
                row = DebugSamename(sa1.source.site, sa1.name, sa2.source.site, sa2.name, str(same_cluster_expected),
                                    common[0], common[1], common[2], top1[0], top1[1], top1[2],top2[0], top2[1], top2[2])
                ds.add_row(row._asdict())

    ds.export_to_csv(_config_.get_output_dir(), 'samename_debug', True)

Res = collections.namedtuple('Res', 'file name id_column detail_columns')

def merge_results(filename, results_to_merge:list):
    edges = []
    sa2data = {}
    name2sa2clusters = {}
    for res in results_to_merge:
        cdo = output_and_gt_adapter.import_cluster_detail_csv(res.id_column, res.file,
                                                        res.detail_columns)
        if res.detail_columns:
            sa2data.update(cdo.sa2other_data)
        sa_orig2clusters = {sa.get_original_attribute(): cls for sa, cls in cdo.sa2clusters.items()}
        name2sa2clusters[res.name] = sa_orig2clusters
        cid2sas = bdsa_utils.multidict_inverter(sa_orig2clusters)
        for sas in cid2sas.values():
            sorted_sas = sorted(sas)
            for sa1, sa2 in itertools.combinations(sorted_sas, 2):
                edges.append(WeightedEdge(sa1, sa2, 1))
    cid2source2sas = collections.defaultdict(bdsa_utils.dd_set_generator)
    cluster_utils.partition_using_agglomerative(edges, cid2source2sas, FreeClusterRules())

    attributes = ['source', 'name'] + ['cid_%s' % name for name in sorted(name2sa2clusters.keys())]
    output_data = dataset.Dataset(attributes)
    for cid, source2sas in cid2source2sas.items():
        for source, sas in source2sas.items():
            for sa in sas:
                row = {'source': source.site, 'name': sa.name, 'cid': cid}
                for name, sa2cls in name2sa2clusters.items():
                    row['cid_%s' % name] = ','.join(sorted(sa2cls.get(sa, [])))
                row.update(sa2data[sa])
                output_data.add_row(row)
    output_data.export_to_csv(_config_.get_output_dir(), 'merged_'+filename, True)

if __name__ == '__main__':
    if argv[1] == 'merge_results':
        files = []
        name_of_file = argv[2]
        for arg in argv[3:]:
            file, name, id_column, detail_column = arg.split(',')
            files.append(Res(file, name, id_column, detail_column.lower() == 'true'))
    merge_results(name_of_file, files)




#ground_truth_path=${root_dir}/functional_tests/expected.csv