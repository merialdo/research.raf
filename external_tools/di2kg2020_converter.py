import collections
import os

from model import dataset
from utils import bdsa_utils, io_utils

from config.bdsa_config import _config_

def create_full_challenge_gt(cat, input_filename):
    gt = dataset.import_csv(input_filename)
    gt_filtered = dataset.Dataset(['source_attribute_id','target_attribute_name'])
    for row in gt.rows:
        if 'target_attribute_name' in row and row['target_attribute_name'].strip() != '':
            gt_filtered.add_row({'source_attribute_id':row['source_attribute_id'],
                                 'target_attribute_name': row['target_attribute_name']})
    gt_filtered.export_to_csv(_config_.get_output_dir(), '%s_di2kg2020' % cat, False)

def create_full_challenge_il(cat, input_filename):
    gt = dataset.import_csv(input_filename)
    gt_filtered = dataset.Dataset(['instance_attribute_id','target_attribute_name'])
    for row in gt.rows:
        if 'target_attribute_name' in row and row['target_attribute_name'].strip() != '' and \
                ('TO_DELETE' not in row or row['TO_DELETE'] != 'yes'):
            gt_filtered.add_row({'instance_attribute_id':row['instance_attribute_id'],
                                 'target_attribute_name': row['target_attribute_name']})
    gt_filtered.export_to_csv(_config_.get_output_dir(), '%s_instance_level_di2kg2020' % cat, False)

def import_csv_linkage_data(cat, linkage_file, dataset_dir, postfix, url_column, id_column,
                            excluded_pids=[]):
    """
    Import a CSV file with linkage to an existing dataset
    :param linkage_file:
    :param dataset_dir:
    :param postfix:
    :param url_column:
    :param id_column:
    :return:
    """
    linkage = dataset.import_csv(linkage_file)
    source2pid2urls = collections.defaultdict(bdsa_utils.dd_set_generator)
    for row in linkage.rows:
        if id_column in row and row[id_column] not in excluded_pids:
            url = row[url_column]
            source = url.split('//')[0]
            source2pid2urls[source][row[id_column]].add(url)
    for source, pid2urls in source2pid2urls.items():
        pid2urls_json = {pid: sorted(urls) for pid, urls in pid2urls.items()}
        filename = ('%s_linkage' % cat) if not(postfix) else  \
            ('%s_linkage_%s' % (cat, postfix))
        directory = os.path.join(dataset_dir, source)
        io_utils.output_json_file(pid2urls_json, filename, directory, False)

def convert_di2kg_il_to_raf(filename):
    gt = dataset.import_csv(filename)
    gt_filtered = dataset.Dataset(['SOURCE_NAME','ATTRIBUTE_NAME', 'URL', 'TARGET_ATTRIBUTE_ID'])
    for row in gt.rows:
        inst_att = row['instance_attribute_id'].split('//')
        url = inst_att[0] + '//' + inst_att[1]
        source_name = inst_att[0]
        att_name = inst_att[2]
        gt_filtered.add_row({
            'SOURCE_NAME': source_name, 'ATTRIBUTE_NAME':att_name,
            'URL': url, 'TARGET_ATTRIBUTE_ID':row['target_attribute_name']
        })
    gt_filtered.export_to_csv(_config_.get_output_dir(), 'raf_gt_il', True)

def convert_di2kg_sm_to_raf(filename):
    gt = dataset.import_csv(filename)
    gt_filtered = dataset.Dataset(['source','name', 'cluster_id_real'])
    for row in gt.rows:
        schema_att = row['source_attribute_id'].split('//')
        source_name = schema_att[0]
        att_name = schema_att[1]
        gt_filtered.add_row({
            'source': source_name, 'name':att_name,
            'cluster_id_real':row['target_attribute_name']
        })
    gt_filtered.export_to_csv(_config_.get_output_dir(), 'raf_gt_sm', True)


if __name__ == '__main__':
    # import_csv_linkage_data('notebook', '/home/federico/projects/benchmark/notebook_linkage_iter3.csv',
    #                         '/home/federico/BDSA/input/dataset/notebook_v2', None, 'spec_id',
    #                         'iter_3_entity_id', ['SINGLET', 'SINGLET?', ''])
    #create_full_challenge_gt('notebook', '/home/federico/BDSA/data/notebook_gt_di2kg.csv')
    #create_full_challenge_il('notebook', '/home/federico/BDSA/data/notebook_instancelevel_di2kg.csv')
    convert_di2kg_sm_to_raf('/home/federico/BDSA/input/ground_truth/notebook_di2kg2020_v2.csv')
    convert_di2kg_il_to_raf('/home/federico/BDSA/input/ground_truth/notebook_instance_level_di2kg2020_v2.csv')