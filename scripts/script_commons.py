import collections

from adapter import adapter_factory
from model import datamodel

UrlValueEntity = collections.namedtuple('UrlValueEntity', 'url value pid add_pid')


def get_sa2urls(only_linked_urls, provide_value=False, provide_pid=False, dataset_name=None, category=None):
    """
    For each SA, get all URLS containing it
    :param: only_linked_urls keep only URLS having some linkage
    :return:
    """
    sa2urls = collections.defaultdict(set)
    linkage = adapter_factory.linkage_factory(random_order=True, dataset_name=dataset_name)
    spec_gen = adapter_factory.spec_factory(dataset_name, category).specifications_generator()
    for source in spec_gen:
        for url, specs in source.pages.items():
            pids = linkage.ids_by_url(url, source.site, source.category)
            main_pid = pids[0] if provide_pid and len(pids) > 0 else None
            other_pids = ','.join(pids[1:]) if provide_pid and len(pids) > 1 else None
            if not only_linked_urls or len(pids) > 0:
                for key in specs.keys():
                    value = specs[key] if provide_value else None
                    res = UrlValueEntity(url, value, main_pid, other_pids)
                    sa2urls[datamodel.source_attribute_factory(source.category, source.site, key)].add(res)
    return dict(sa2urls)
