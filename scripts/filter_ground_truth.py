"""
Allows filtering and modifying ground truth
"""
import random

import collections

import itertools

from adapter import output_and_gt_adapter
from model import dataset, datamodel
from model.datamodel import Provenance
from model.dataset import Dataset
from scripts.results_evaluation import ResultsEvaluator
from utils import bdsa_utils
from config.bdsa_config import _config_

SchemaGTRow = collections.namedtuple('SchemaGTRow', 'source_attribute_id target_attribute_name')

InstancePairsGTRow = collections.namedtuple('InstanceGTRow',
                                       'attribute_instance_left attribute_instance_right')

InstanceGTRow = collections.namedtuple('InstanceGTRow',
                                       'instance_attribute_id target_attribute_name')

def srbuild(sa, ta):
    return SchemaGTRow('%s//%s' % (sa.source.site, sa.name), ta)

def irbuild(instance, ta):
    str_instance = '%s//%s//%s' % (
        instance.source.site, instance.url.split('/')[-1], instance.sa.name)
    return InstanceGTRow(str_instance, str(ta))


def irpbuild(left, right, val):
    left_str = '%s/%s/%s=%s' % (
            left.source.site, left.url.split('/')[-1], left.sa.name, left.value)
    right_str = '%s/%s/%s=%s' % (
        right.source.site, right.url.split('/')[-1], right.sa.name, right.value)
    return InstancePairsGTRow(left_str, right_str)

def sample_schema_alignment_ground_truth(ratio=0.1):
    """
    Filter schema alignment ground truth
    For each possible group of TA, takes X% of assigned SA (minimum 1)
    :return: 
    """
    evaluator = ResultsEvaluator()
    gt_schema = evaluator.schema_gt
    heterogeneous = evaluator.instance_gt.heterogeneous_sa
    rows_filtered = _sample_sagt_intern(ratio, gt_schema.sa2clusters, heterogeneous)
    ds = Dataset(SchemaGTRow._fields)
    ds.add_rows([arow._asdict() for arow in rows_filtered])
    ds.export_to_csv(_config_.get_output_dir(),
                     'filtered_schemagt_%s' % _config_.get_category(), True)

def _sample_sagt_intern(ratio, sa2clusters:dict, heterogeneous):
    """
    Internal method for sample_schema_alignment_ground_truth
    :param ratio: 
    :param sa2clusters: 
    :return: 
    """
    # Build dict associating each TA to SAS it provides
    ta_group2sas = bdsa_utils.build_dict(sa2clusters.items(),
                          key_getter=lambda sa2ta_pair: frozenset(sa2ta_pair[1]),
                          value_getter=lambda sa2ta_pair: sa2ta_pair[0])

    all_tas = set().union(*sa2clusters.values())
    # At least 50% of possible groups, keeping in mind that every TA should be covered
    tagr_filt2sas = {}
    min_groups = int(round(len(ta_group2sas) / 2))
    ta_groups_available = set(ta_group2sas.keys())
    tas_missing = set(all_tas)

    while len(tagr_filt2sas) < min_groups or len(tas_missing) > 0:
        if len(tas_missing) > 0:
            group = random.choice([tagr for tagr in ta_groups_available if len(tagr & tas_missing) > 0])
        else:
            group = random.choice(sorted(ta_groups_available))
        ta_groups_available.remove(group)
        tagr_filt2sas[group] = ta_group2sas[group]
        tas_missing.difference_update(group)

    tagr2sas_filtered  = {}
    # For each TA group, select a sample of SAS
    for ta_group, sas in tagr_filt2sas.items():
        # For each group, we select ratio% (min 1)
        nb_sas = max(1, round(len(sas) * ratio * 1.1))
        sas_filtered = random.sample(sas, nb_sas)
        tagr2sas_filtered[ta_group] = sas_filtered
    # Build output, in form of rows with each association SA-TA
    rows_result = []
    sasses = set()
    for ta_group, sas_filter in tagr2sas_filtered.items():
        for ta_sa in itertools.product(ta_group, sas_filter):
            rows_result.append(srbuild(ta_sa[1], ta_sa[0]))
            sasses.add(ta_sa[1])
    total_nb_sas = len(sa2clusters.keys())
    ratio_heterogeneous = len(heterogeneous) / total_nb_sas
    min_heterogeneous = max(3, round(int(len(sasses) * ratio_heterogeneous)))
    while len(sasses & heterogeneous) < min_heterogeneous:
        sa_new = random.choice(list(heterogeneous - sasses))
        for cluster in sa2clusters[sa_new]:
            rows_result.append(srbuild(sa_new, cluster))
        sasses.add(sa_new)
    return rows_result

def sample_instance_level_tas(filtered_schema_level, ratio=0.1, output_result=True):
    """
    Create a sample of instance-level, in form of association IL-target attribute
    :param ratio: 
    :param filtered_schema_level: the schema level data already sampled
    :return: 
    """
    evaluator = ResultsEvaluator()
    instance_gt = evaluator.instance_gt
    schema_filtered = dataset.import_csv(filtered_schema_level)
    rows = _sample_instance_level_tas_internal(
        instance_gt.prov2ta, schema_filtered, ratio)
    ds = Dataset(InstanceGTRow._fields)
    ds.add_rows([arow._asdict() for arow in rows])
    ds.export_to_csv(_config_.get_output_dir(),
                     'filtered_instancegt_%s' % _config_.get_category(), True)


def _sample_instance_level_tas_internal(prov2tas:dict, schema_gt:Dataset, ratio):
    sas_associations = collections.defaultdict(set)
    # We first check for all sas in sampled schema level data
    for el in schema_gt.rows:
        sa_parts = el['source_attribute_id'].split('//')
        sa = datamodel.source_attribute_factory(_config_.get_category(),
                                                sa_parts[0], sa_parts[1])
        sas_associations[sa].add(el['target_attribute_name'])
    output_ds = []
    sas_association_instances = collections.defaultdict(set) # This is just for a check
    sa2tas2provs = collections.defaultdict(bdsa_utils.dd_set_generator)
    for prov, tas in prov2tas.items():
        if prov.sa in sas_associations:
            sa2tas2provs[prov.sa][frozenset(tas)].add(prov)
            sas_association_instances[prov.sa].update(tas)
    # Check coherence
    for sa, tas_insta in sas_association_instances.items():
        tas_schema = sas_associations[sa]
        if tas_insta != tas_schema:
            print('!!! %s: %s - %s' % (str(sa), str(tas_insta), str(tas_schema)))
    for sa, tas2provs in sa2tas2provs.items():
        for tas, provs in tas2provs.items():
            minimum = min(3, len(provs))
            provs_filtered = random.sample(
                provs, max(minimum, int(round(ratio * len(provs)))))
            for prov in provs_filtered:
                for ta in tas:
                    output_ds.append(irbuild(prov, ta))
    return output_ds

def sample_instance_level_ground_truth(ratio=0.1):
    """
    Sample the instance level GT, providing pair of instances in linkage:
    * with exactly the same group of TA
    * with some common TA
    :param ratio: 
    :return: 
    """
    evaluator = ResultsEvaluator()
    instance_gt = evaluator.instance_gt
    row_pairs = _sample_ilgt_intern(ratio, instance_gt.prov2ta)
    ds = Dataset(InstancePairsGTRow._fields)
    ds.add_rows([arow._asdict() for arow in row_pairs])
    ds.export_to_csv(_config_.get_output_dir(),
                     'filtered_instancegt_%s' % _config_.get_category(), True)


def _sample_ilgt_intern(ratio, prov2ta:dict):
    # Build dict associating each TA to PROVS it provides
    ta_group2provs_set = bdsa_utils.build_dict(prov2ta.items(),
                          key_getter=lambda prov2ta_pair: frozenset(prov2ta_pair[1]),
                          value_getter=lambda prov2ta_pair: prov2ta_pair[0])
    ta_group2provs_list = {tagr: sorted(provs) for tagr, provs in ta_group2provs_set.items()}
    row_pairs = []
    # Build pair with instance attribute from same group of TA
    # Select Ratio% of single instances and, on these, len*2 random edges
    # (over len*len-1 /2)
    for prov_list in ta_group2provs_list.values():
        if len(prov_list) <=1:
            continue
        # For efficiency, random pair of provs are built on indexes and not on actual provs
        nb_samples = max(2, int(round(len(prov_list) * ratio)))
        sample_provs = random.sample(prov_list, nb_samples)
        nb_pair_samples = nb_samples * 2 if nb_samples >= 5 \
            else int(nb_samples * (nb_samples - 1) / 2)
        sample_indexes = random.sample(list(itertools.combinations(
            range(nb_samples), 2)), nb_pair_samples)
        for indexes in sample_indexes:
            row_pairs.append(irpbuild(sample_provs[indexes[0]],
                                      sample_provs[indexes[1]], 'in'))

    # Now we build pairs of TA groups that share some TAs
    # First we need to detect those groups
    pairs_ta_groups = _detect_pair_tagroup_sharing_atleast_one_ta(ta_group2provs_set)
    pairs_ta_groups_filtered = random.sample(pairs_ta_groups, int(round(len(pairs_ta_groups) * ratio * 2)))
    for pair in pairs_ta_groups_filtered:
        provs1 = ta_group2provs_list[pair[0]]
        provs2 = ta_group2provs_list[pair[1]]
        provs1_sampled = random.sample(provs1, max(1, int(round(len(provs1) * ratio))))
        provs2_sampled = random.sample(provs2, max(1, int(round(len(provs2) * ratio))))
        nb_samples = min(
                len(provs1_sampled) * len(provs2_sampled),
                max(len(provs1_sampled), len(provs2_sampled)) * 2)
        sample_indexes = random.sample(list(itertools.product
                                (range(len(provs1_sampled)), range(len(provs2_sampled)))),
                                       nb_samples)
        for indexes in sample_indexes:
            row_pairs.append(irpbuild(provs1_sampled[indexes[0]],
                                      provs2_sampled[indexes[1]], 'out'))
    return row_pairs


def _detect_pair_tagroup_sharing_atleast_one_ta(ta_group2provs_set):
    """
    Return all pairs of groups of target attribute with at least a common ta
    If they share more than one, they will still appear once
    :param ta_group2provs_set: 
    :return: 
    """
    ta2ta_group = collections.defaultdict(set)
    for ta_group in ta_group2provs_set.keys():
        for ta in ta_group:
            ta2ta_group[ta].add(ta_group)
    pairs_ta_groups = set()
    for ta_groups in ta2ta_group.values():
        ta_groups_listed = sorted(ta_groups)
        pairs_ta_groups.update(itertools.combinations(ta_groups_listed, 2))
    return pairs_ta_groups

def remove_useless_rows_columns(cat, filename):
    gt = dataset.import_csv(filename)
    gt_filtered = dataset.Dataset(['source','name','cluster_id_real'])
    for row in gt.rows:
        if 'target_attribute_name' in row and row['target_attribute_name'].strip() != '':
            gt_filtered.add_row({'source':row['source'], 'name':row['attribute_name'],
                                 'cluster_id_real': row['target_attribute_name']})
    gt_filtered.export_to_csv(_config_.get_output_dir(), cat, False)



if __name__ == '__main__':
    #sample_instance_level_ground_truth(0.05)
    #sample_schema_alignment_ground_truth(0.15)
    sample_instance_level_tas(
          '/home/federico/BDSA/data/notebook_labelled_data_schema_matching_di2kg2020.csv',
                               0.2, False)
