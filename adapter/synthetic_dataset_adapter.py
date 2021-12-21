"""
This file contains some classes and methods useful to interface with data built with the synthetic dataset builder.
"""
import collections
import re
from types import FunctionType

from model import datamodel

SOURCE_SEPARATOR = '--s--'


class SyntheticSourceAttribute:
    """
    Synthetic source attributes have a specific encoding that allows to distinguish their features.
    Here we extract them.
    
    >>> sa = SyntheticSourceAttribute(datamodel.source_attribute_factory('cat', \
        'www.LIPsiOsCqlhOQYh--s--H--s--10--s--100--s--700.com', 'gJCivH7@@@H@2@0,100000@15-0'))
    >>> (sa.synth_source.site, sa.synth_source.ht, sa.synth_source.value_error, \
    sa.synth_source.linkage_error, sa.synth_source.linkage_missing)
    ('www.LIPsiOsCqlhOQYh.com', 'H', 0.01, 0.1, 0.7)
    >>> (sa.attname, sa.att_ht, sa.cardinality, sa.error_rate)
    ('gJCivH7', 'H', 2, 0.1)
    """

    def __eq__(self, o: object) -> bool:
        return (self.synth_source, self.attname) == (o.synth_source, o.attname)

    def __hash__(self) -> int:
        return hash((self.synth_source, self.attname))

    def __init__(self, _sa: datamodel.SourceAttribute):
        self.synth_source = SyntheticSource(_sa.source)
        attribute_search = re.search('([a-zA-Z0-9]+)@@@(H|T)@([0-9]+)@(\d+,\d+)@(\d+)-(\d+)',
                                     _sa.name)
        if attribute_search:
            self.attname = attribute_search.group(1)
            self.att_ht = attribute_search.group(2)
            self.cardinality = int(attribute_search.group(3))
            self.error_rate = float(attribute_search.group(4).replace(',', '.'))
        else:
            self.attname = _sa.name
            self.att_ht = '?'
            self.cardinality = 0
            self.error_rate = 0

class SyntheticPage:
    """
    Synthetic page: their source have a specific encoding that allows to distinguish their features.
    Here we extract them.
    
    >>> spage = SyntheticPage(datamodel.page_factory('www.LIPsiOsCqlhOQYh--s--H--s--10--s--100--s--700.com/385/', \
        datamodel.SourceSpecifications("www.LIPsiOsCqlhOQYh--s--H--s--10--s--100--s--700.com", 'cat', None)))
    >>> (spage.synth_source.site, spage.synth_source.ht, spage.synth_source.value_error, \
    spage.synth_source.linkage_error, spage.synth_source.linkage_missing)
    ('www.LIPsiOsCqlhOQYh.com', 'H', 0.01, 0.1, 0.7)
    >>> (spage.url, spage.pid)
    ('www.LIPsiOsCqlhOQYh--s--H--s--10--s--100--s--700.com/385/', '385')
    """

    def __eq__(self, o: object) -> bool:
        return (self.synth_source, self.url) == (o.synth_source, o.url)

    def __hash__(self) -> int:
        return hash((self.synth_source, self.url))

    def __init__(self, _page: datamodel.Page):
        self.synth_source = SyntheticSource(_page.source)
        self.pid = _page.url.split('/')[1]
        self.url = _page.url



class SyntheticSource:
    """
    Synthetic source have a specific encoding that allows to distinguish their features.
    Here we extract them.
    
    >>> source = SyntheticSource(datamodel.SourceSpecifications("www.LIPsiOsCqlhOQYh--s--H--s--10--s--100--s--700.com",\
     'cat', None))
    >>> (source.site, source.ht, source.value_error, source.linkage_error, source.linkage_missing)
    ('www.LIPsiOsCqlhOQYh.com', 'H', 0.01, 0.1, 0.7)
    
    
    """

    def __eq__(self, o: object) -> bool:
        return self.site == o.site

    def __hash__(self) -> int:
        return hash(self.site)

    def __init__(self, _source: datamodel.SourceSpecifications):
        url_parts = _source.site.split('.')
        domain_with_features = url_parts[1].split(SOURCE_SEPARATOR)
        self.ht = domain_with_features[1]
        self.value_error = int(domain_with_features[2]) / float(1000)
        self.linkage_error = int(domain_with_features[3]) / float(1000)
        self.linkage_missing = int(domain_with_features[4]) / float(1000)
        self.site = '.'.join((url_parts[0], domain_with_features[0], url_parts[2]))


def golden_set_for_synthetic_data(nid2source2nodes: dict, isolated: list, cluster_id_getter: FunctionType,
                                  synthetic_builder: FunctionType) -> (list, list):
    """
    Convert nodes into a synthetic object AND return 2 clustered results:
    - according to current clustering (so just as in nid2source2nodes just removing the ID and flattening source2nodes)
    - according to name/url (as in synthetic dataset atts with same name are equivalent)
    :param nid2source2nodes:
    :param isolated: 
    :param cluster_id_getter: how the get the name/url 
    :param synthetic_builder: how to build the synthetic element
    
    :return: 
    """
    expected_clusters = collections.defaultdict(list)
    computed_clusters = []
    for source2sas in nid2source2nodes.values():
        comp_cluster = []
        for sas in source2sas.values():
            for sa in sas:
                synthetic_element = synthetic_builder(sa)
                comp_cluster.append(synthetic_element)
                expected_clusters[cluster_id_getter(synthetic_element)].append(synthetic_element)
        computed_clusters.append(comp_cluster)
    for sa in isolated:
        synthetic_isolated_element = synthetic_builder(sa)
        expected_clusters[cluster_id_getter(synthetic_isolated_element)].append(synthetic_isolated_element)
        computed_clusters.append([synthetic_isolated_element])
    return expected_clusters, computed_clusters

if __name__ == "__main__":
    import doctest
    doctest.testmod()