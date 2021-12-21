import json
import os
import re
import string

from model import datamodel
from utils import string_utils

SPACE_TOKEN = '__s__'


def rename_column(name, source):
    return 'http://%s/%s' % (source, name.replace(' ', SPACE_TOKEN))


def t2k_att2sa(t2k_att, category):
    reg = re.search('http://([a-zA-Z0-9.-]+)/([a-zA-Z0-9_]+)', t2k_att)
    att_name_old = reg.group(2)
    att_name_fixed = att_name_old.replace(SPACE_TOKEN, ' ')
    sa = datamodel.source_attribute_factory(category, reg.group(1), att_name_fixed)
    return sa


def import_json_source(category, source_directory):

    # Import all data
    spec_file = os.path.join(source_directory, '%s_spec.json' % category)
    linkage_file = os.path.join(source_directory, '%s_linkage.json' % category)

    # Normalize and remove page title
    with open(spec_file) as camera_spec:
        camera_spec_data = json.load(camera_spec)
        for camera_spec in camera_spec_data:
            specs = camera_spec['spec']
            camera_spec['url'] = string_utils.url_normalizer(camera_spec['url'])
            for key in list(specs.keys()):
                value = specs[key]
                del specs[key]
                if key != '<page title>': #remove page title
                    #new_key = re.sub('[^a-zA-Z]+', '_', key)
                    #new_value = re.sub('[^a-zA-Z0-9]+', '_', value)
                    new_key = string_utils.folding_using_regex(key)
                    specs[new_key] = normalize_value(value)
    with open(linkage_file) as camera_linkage:
        camera_linkage_data = json.load(camera_linkage)
        for key in list(camera_linkage_data.keys()):
            camera_linkage_data[key] = [string_utils.url_normalizer(url) for url in camera_linkage_data[key]]
    return camera_linkage_data, camera_spec_data


def normalize_value(old_value):
    new_value = string_utils.normalize_keyvalues(old_value)
    return new_value.replace('"', '').replace('\n', '').replace(
        '\\', '').replace('\"', '').strip()

#T2K does not manage correctly numeric or mostly-numeric IDs
NUMBERS2LETTERS = dict(zip([str(x) for x in range(10)], string.ascii_uppercase[:10]))

def get_entity_label(current_url, url2linkage, source=''):
    """
    Get ENTITY ID of url
    """
    labels = url2linkage.get(current_url, {})
    if len(labels) == 0:
        url_to_find_components = current_url.split("/")
        return "NO_" + source + '_' + url_to_find_components[len(url_to_find_components) - 1]
    if len(labels) > 1:
        print('******* WARNING: MULTIPLE LABELS FOR URL: %s ***' % str(labels))
    first_label = ''.join(NUMBERS2LETTERS.get(x, x) for x in list(labels)[0])
    return 'E-%s' % first_label