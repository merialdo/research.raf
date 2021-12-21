# Some methods that simplify writing tests and doc tests
from model import datamodel
from model.datamodel import SourceSpecifications


def tsa(att_name, site='test_site', category='dummy'):
    """
    Build a source attribute
    :param source_name:
    :param att_name:
    :return:
    """
    return datamodel.source_attribute_factory(category, site, att_name)

def ts(site='test_site', category='dummy'):
    """
    Build a source attribute
    :param source_name:
    :param att_name:
    :return:
    """
    return datamodel.SourceSpecifications(site, category, None)

class TspBuilder:
    def __init__(self, site, category):
        self.site = site
        self.category = category
        self.pages = {}
        self.current_page = {}

    def kv(self, k='defk', v='defv'):
        self.current_page[str(k)] = str(v)
        return self

    def p(self, url):
        self.pages[url] = self.current_page
        self.current_page = {}
        return self

    def end(self):
        return SourceSpecifications(self.site, self.category, self.pages)

def tsp(site='test_site', category='dummy'):
    return TspBuilder(site, category)

def tp(name, site, pid):
    return datamodel.provenance_factory(site, 'dummy', pid, name, value='')