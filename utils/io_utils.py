# coding=utf-8
import bisect
import collections
import json
import os
import csv

#import dill as pickle
import pickle
#import msgpack
import pandas
from tqdm import tqdm

from config.bdsa_config import _config_
from utils import string_utils
import hashlib

SER = 'ser'

JSON = 'json'
CSV = 'csv'
TXT = 'txt'

FilePath = collections.namedtuple('FilePath', 'name path')

def browse_directory_files(directory_path, filter=None):
    """
    Browse directory files according to a given filter function
    :param directory_path:
    :param filter:
    :return:
    """
    files = []
    for file_name in os.listdir(directory_path):
        if not filter or filter(file_name):
            file_path = os.path.join(directory_path, file_name)
            bisect.insort(files, FilePath(file_name, file_path))
    return files

def output_json_file(object, filename, directory =_config_.get_output_dir(), timestamp=True, sort_keys=True):
    with open(_build_filename(filename, JSON, directory, timestamp), 'w', encoding='utf-8') as outfile:
        json.dump(object, outfile, indent=4, sort_keys=sort_keys, default=lambda x: x.__dict__)

def import_json_file(filename):
    with open(filename, 'r', encoding="utf8") as infile:
        res = json.load(infile)
    return res

MAX_BLOCK_SIZE = 65536
def compute_file_hash(filename):
    hasher = hashlib.md5()
    with open(filename, 'rb') as afile:
        buf = afile.read(MAX_BLOCK_SIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(MAX_BLOCK_SIZE)
    return hasher.hexdigest()

def get_or_create_cache_file(id, instance_id, create_function):
    """
    Load cache file if exists, build it otherwise
    :param id: identifier of the cache file to create
    :param instance_id: second identifier, can be used for instance to distinguish between cache of sample or full dataset
    :return: 
    """
    filename = _cache_file_name(id, instance_id)
    if os.path.exists(filename):
        with open(filename, 'rb') as infile:
            res = pickle.load(infile)
    else:
        res = create_function()
        with open(filename, 'wb') as outfile:
            pickle.dump(res, outfile)
    return res


def _cache_file_name(id, name):
    name = name.replace('/', '___')
    return os.path.join(_config_.get_cache_dir(), name + '__' + id + '.ser')

def quick_csv_single(data, filename, asdict=False, _fieldnames=None):
    if asdict:
        data = data._asdict()
    with open(filename, 'w', encoding='utf8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_fieldnames or data.keys(),extrasaction='ignore')
        writer.writeheader()
        writer.writerow(data)

def quick_csv_group(data, filename, asdict=False, fieldnames=None):
    if fieldnames is None:
        fieldnames = set()
        for el in data:
            fieldnames.update(el._asdict().keys() if asdict else el.keys())
    with open(filename, 'w', encoding='utf8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames, extrasaction='ignore')
        writer.writeheader()
        for element in data:
            if asdict:
                element = element._asdict()
            writer.writerow(element)

def output_csv_file(array, filename, directory = _config_.get_output_dir(), timestamp=True):
    with open(_build_filename(filename, CSV, directory, timestamp), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(array)

def append_generic_file(filename, text, add_newline=True):
    """
    Append a line to a generic file
    :param filename:
    :param text:
    :return:
    """
    with open(filename, "a") as myfile:
        myfile.write(text+'\n' if add_newline else text)

def append_csv_file(fields, list_data, filename, directory = _config_.get_output_dir()):
    """
    Append data to a CSV file
    :param fields:
    :param data:
    :param filename:
    :param directory:
    :return:
    """
    csv_file = os.path.join(directory, filename)
    already_created = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields,
                                extrasaction='ignore')
        if not already_created:
            writer.writeheader()
        for data in list_data:
            writer.writerow(data)

def output_txt_file(array, filename, directory = _config_.get_output_dir(), timestamp=True):
    with open(_build_filename(filename, TXT, directory, timestamp), 'w') as f:
        for k, v in array.items():
            f.write("%s --> %s\n\n"%(str(k), str(v)))

def import_generic_file_per_line(filepath, remove_line_break=False):
    """
    Import a file line per line
    :param filepath:
    :return:
    """
    with open(filepath, 'r') as f:
        for line in f:
            if remove_line_break:
                line = line.strip('\n\r')
            if line != '':
                yield line

def build_directory_output(dir_name:str, path = _config_.get_output_dir(), timestamp=True):
    dir_path = _build_filename(dir_name, None, path, timestamp)
    os.mkdir(dir_path)
    return dir_path

def output_file_generic(objects, converter, filename, directory = _config_.get_output_dir(), timestamp=True, extension=TXT):
    """
    Print a file given a list of objects and a converter
    :param objects:
    :param converter:
    :param filename:
    :param directory:
    :param timestamp:
    :param extension:
    :return:
    """
    with open(_build_filename(filename, extension, directory, timestamp), 'w', encoding='utf-8') as f:
        f.writelines(converter(obj)+'\n' for obj in objects)

def _build_filename(filename, extension, directory = _config_.get_output_dir(), timestamp=True):
    """
    Outputs a complete path of output file, adding standard output dir, extension and timestamp
    :param filename: 
    :param extension: 
    :return: 
    """
    timestamp_string = '_' + string_utils.timestamp_string_format() if timestamp else ''
    extension_postfix = '.' + extension if extension else ''
    return os.path.join(directory, filename + timestamp_string + extension_postfix)

if __name__ == '__main__':
    output_csv_file([['giàààào', 'hssss'], ['piàno', 'he']], 'giao')


def find_files_pattern(directory, prefix,  postfix, extension):
    """
    Find all files in provided directory that have a provided prefix.
    Text until given postfix will be returned, also with filename
    :param extension: 
    :param directory: 
    :param prefix: 
    :param postfix: 
    :return: 
    """
    key2file = {}
    for filename in os.listdir(directory):
        name, current_extension = os.path.splitext(filename)
        #Is extension the good one? Or if no extension asked has really the file no extensions?
        ok_extension = (extension and current_extension == '.' +extension) or (not extension and current_extension == '')
        if ok_extension and name.startswith(prefix):
            key = name.replace(prefix, '').split(postfix)[0]
            key2file[key] = os.path.join(directory, filename)
    return key2file

def import_ser_file(filename):
    with open(filename, 'rb') as input_file:
        res = pickle.load(input_file)
    return res


def output_ser_file(dto, filename, timestamp=True):
    with open(_build_filename(filename, SER, _config_.get_cache_dir(), timestamp), 'wb') as outfile:
        pickle.dump(dto, outfile)

def build_dict_from_csv(input_csv_file, key_row, value_build, multi=False):
    output_dict = collections.defaultdict(set) if multi else {}
    with open(input_csv_file) as csvfile:
        data = csv.DictReader(csvfile)
        for row in data:
            if multi:
                output_dict[row[key_row]].add(value_build(row))
            else:
                output_dict[row[key_row]] = value_build(row)
    return {x: sorted(v) for x, v in output_dict.items()} if multi else output_dict