import collections
import math

from adapter import adapter_factory
from config.bdsa_config import _config_
import pandas as pd

# Goal: evaluate percentage of available linkages


def _compute_size_entities():
    """
    Compute size of each entity (=nb of pages associated to a single product ID)
    """
    linkage = adapter_factory.linkage_factory()
    sgen = adapter_factory.spec_factory().specifications_generator()

    pid2pages = collections.defaultdict(list)

    total_nb_pages = 0
    for source in sgen:
        total_nb_pages += len(source.pages)
        for url in source.pages:
            page_ids = linkage.ids_by_url(url, source.site, source.category)
            for pid in page_ids:
                pid2pages[pid].append(url)
    size_of_entities = sorted([len(pages) for pages in pid2pages.values()], reverse=True)
    total_pages_in_linkage = sum(size_of_entities)
    ratio = total_nb_pages / total_pages_in_linkage
    print("Situation: %d total, %d total linkage, %.2f ratio" % (total_nb_pages, total_pages_in_linkage, ratio))
    return size_of_entities, ratio


def _compute_buckets_log(step:int):
    """
    Compute buckets of entity sizes AND nb of pairs in linkage
    """
    size_entities, ratio = _compute_size_entities()

    # Now compute nb of pairs in linkage
    nb_linkage_pairs = sum([x * (x - 1) / 2 for x in size_entities])

    buckets = {}
    i = 0
    low_interval = 0
    nb_instances = 0
    while nb_instances < len(size_entities):
        max_interval = 10 ** (i * step)
        nb_elements = len([x for x in size_entities if low_interval <= x < max_interval])
        nb_instances += nb_elements
        buckets[max_interval] = nb_elements
        i += 1
        low_interval = max_interval
    #print(','.join(str(x[1]) for x in sorted(buckets.items(), key=lambda x: x[0])))
    print(str(buckets))
    return buckets, ratio, nb_linkage_pairs


def estimate_linkage_portion():
    buckets, ratio_pages_linkage, nb_linkage_pairs = _compute_buckets_log(0.25)
    buckets_adapted = {key: value * ratio_pages_linkage for key, value in buckets.items()}

    # now we estimate nb of pairs
    low = 1
    total_nb_pairs_estimated = 0
    for i in sorted(buckets_adapted.keys()):
        avg_size = math.sqrt(i * low)
        total_nb_pairs_estimated += avg_size * (avg_size - 1) / 2 * buckets_adapted[i]
        low = i

    total_pairs_orig = 0
    for i in sorted(buckets.keys()):
        avg_size = math.sqrt(i * low)
        total_pairs_orig += avg_size * (avg_size - 1) / 2 * buckets[i]
        low = i
    print("estimation %.2f, real %d" % (total_pairs_orig, nb_linkage_pairs))
    print("nb of linkages actual %.2f, linkages "
          "estimated %.2f, ratio %.2f" % (nb_linkage_pairs, total_nb_pairs_estimated,
                                            100 * nb_linkage_pairs / total_nb_pairs_estimated))



    # Now we compute nb of pairs that we have



estimate_linkage_portion()
