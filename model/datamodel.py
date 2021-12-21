import collections

from cachetools import cached, LRUCache

from config import constants
from utils import string_utils

__author__ = 'fpiai'

## SOURCE ##
BUCKET_SIZE_PROVS = 10
SourceSpecificationsBase = collections.namedtuple("SourceSpecifications", "site category pages")

class SourceSpecifications(SourceSpecificationsBase):
    def __str__(self):
        return self.site + '__' + self.category

    def metadata_only(self):
        """
        Return the source without the specifications (useful to save space)
        :return: 
        """
        return SourceSpecifications(self.site, self.category, None)

    def __eq__(self, other):
        """
        Note that the equals ignores the specifications associated, looks only for site and category
        :param other: 
        :return: 
        """
        return self.site == other.site and self.category == other.category

    def __hash__(self):
        return hash((self.category, self.site))

    def __lt__(self, other):
        if self.site == other.site:
            return self.category < other.category

        return self.site < other.site




## SOURCE ATTRIBUTE ##
SourceAttributeBase = collections.namedtuple("SourceAttribute", "source name")

class SourceAttribute(SourceAttributeBase):
    def __str__(self):
        return str(self.source) + '/' + self.name

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.source == other.source and self.name == other.name
        return False

    def __hash__(self):
        return hash((self.source, self.name))

    def __lt__(self, other):
        if self.source == other.source:
            return self.name < other.name

        return self.source < other.source

    def get_site(self):
        return self.source.site

    def get_category(self):
        return self.source.category

    def is_generated(self):
        return constants.GENERATED_ATTS_SEPARATOR in self.name

    def get_original_name(self):
        return self.name.split(constants.GENERATED_ATTS_SEPARATOR)[0]

    def get_original_attribute(self):
        return source_attribute_factory(self.source.category, self.source.site, self.get_original_name())

def source_attribute_factory(category:str, site:str, name:str, generated_suffix=None):
    """
    Source attribute factory
    :param category: 
    :param site: 
    :param name: 
    :return: 
    """

    source = SourceSpecifications(site, category, None)
    complete_name = '%s%s%s' % (name, constants.GENERATED_ATTS_SEPARATOR, generated_suffix) if generated_suffix else name
    return SourceAttribute(source, complete_name)

PageBase = collections.namedtuple("PageBase", "source url")
## PAGE ##
class Page(PageBase):

    def __str__(self):
        return '%s/%s' % (str(self.source), self.url)

    def __eq__(self, other: object) -> bool:
        if isinstance(self, other.__class__):
            return self.url == other.url and self.source == other.source
        return False

    def __hash__(self) -> int:
        return hash((self.source, self.url))

    def __lt__(self, other):
        if self.source == other.source:
            return self.url < other.url

        return self.source < other.source

    @property
    def name(self):
        """
        Alias for url
        :return: 
        """
        return self.url

def page_factory(url, source: SourceSpecifications):
    """
    Factory of a page
    :param url: 
    :param site: 
    :param category: 
    :return: 
    """
    return Page(source.metadata_only(), url)

def page_factoryz(url, site, cat='cat'):
    return page_factory(url, SourceSpecifications(site, cat, None))


class Provenance:
    """
    Provenance in Knowledge Graph
    """
    def __init__(self, url, sa, value):
        self.url = url
        self.sa = sa
        self.value = value
        self.hash = hash((self.sa.name, self.sa.source.site, self.url))
        self.bucket = self.hash % BUCKET_SIZE_PROVS

    def __str__(self):
        return '%s#-#%s/%s=%s' % (self.sa.source.site, self.url, self.sa.name, self.value)

    def __repr__(self):
        return str(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(self, other.__class__):
            return self.url == other.url and self.sa == other.sa
        return False

    def __hash__(self) -> int:
        return self.hash

    def __lt__(self, other):
        if self.url == other.url:
            return self.sa < other.sa

        return self.url < other.url

    @property
    def source(self):
        return self.sa.source

def provenance_factory(site, category, url, name, value=''):
    sa = source_attribute_factory(category, site, name)
    return Provenance(url, sa, value)

def convert_provenance(prov:str, cat='dummy') -> Provenance:
    """
    Convert a provenance in IKGPP
    :param prov:
    :return:
    """
    if '#-#' in prov:
        site2rest = prov.split('#-#')
        site = site2rest[0]
        prov = site2rest[1]
    else: #Legacy
        site = prov.replace('http:/','').lstrip('/').split('/')[0]

    namevalue = prov.split('=')
    value = namevalue[1]
    prov_noval = namevalue[0]
    url = '/'.join(prov_noval.split('/')[:-1])
    name = prov_noval.split('/')[-1]
    return Provenance(url, source_attribute_factory(cat, site, name), value)