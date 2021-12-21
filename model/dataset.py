import collections
import csv
import os
from utils import string_utils


class Dataset(collections.namedtuple('Dataset', ['ud_headers', 'discovered_headers', 'rows'])):
    """
    Dataset of elements, in form of list of dict attr-->value.
    Can be imported/exported in csv files.
    Column headers may be provided by users in ud_headers parameters.
    If some attributes in rows are not found in the user-provided headers, additional headers
    will be added as columns after all the ud headers, alphabetically sorted.
    
    
    """
    def __new__(cls, _ud_headers=[]):
        self = super(cls, Dataset).__new__(cls, _ud_headers, set([]), [])
        return self

    def add_row(self, row):
        self.rows.append(row)
        self.discovered_headers.update(row.keys())

    def add_rows(self, rows):
        for row in rows:
            self.add_row(row)

    def add_element_to_header(self, element):
        self.ud_headers.append(element)
        self.discovered_headers.add(element)

    def export_to_csv(self, directory, name, with_timestamp, extension='csv', delimiter=','):
        """
        Export to provided CSV
        :param directory: 
        :param name: 
        :param with_timestamp: 
        :return: 
        """
        timestamp = '_'+string_utils.timestamp_string_format() if with_timestamp else ''
        csv_file = os.path.join(directory, name + timestamp + '.'+extension)
        headers = list(self.ud_headers)
        other_discovered_headers = self.discovered_headers.difference(self.ud_headers)
        headers.extend(sorted(list(other_discovered_headers)))
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile,
                                    fieldnames=headers,
                                    extrasaction='ignore', delimiter=delimiter)
            writer.writeheader()
            writer.writerows(self.rows)


def import_csv(filepath, ud_headers=None):
    with open(filepath, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        headers = ud_headers or reader.fieldnames
        result = Dataset(headers)
        for row in reader:
            result.add_row(row)
    return result
