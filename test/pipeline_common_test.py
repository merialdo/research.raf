import collections
import unittest
from warnings import catch_warnings

import pipeline.cluster_utils
from model import datamodel
from model.clusters import SaClusterRules
from pipeline import pipeline_common
from pipeline.cluster_utils import WeightedEdge
from utils import bdsa_utils


class TestPipelineAnalyzer(unittest.TestCase):
    def test_clustering(self):
        cat = 'category'
        s1a1a = datamodel.source_attribute_factory(cat, 1, 's1.a1a')
        s1a1b = datamodel.source_attribute_factory(cat, 1, 's1.a1b')
        s1a1c = datamodel.source_attribute_factory(cat, 1, 's1.a1c')
        s1a2 = datamodel.source_attribute_factory(cat, 1, 's1.a2')
        s2a1 = datamodel.source_attribute_factory(cat, 2, 's2.a1')
        s2a2 = datamodel.source_attribute_factory(cat, 2, 's2.a2')

        edges = [WeightedEdge(s1a1a, s2a1, 1), WeightedEdge(s1a1b, s2a1, 1),
                 WeightedEdge(s1a1c, s2a1, 0.8),  WeightedEdge(s1a2, s2a1, 0.8), WeightedEdge(s1a2, s2a2, 1)]

        sa2urls = {s1a1a: set([1, 2]), s1a1b: set([3, 4]), s1a1c: set([5]), s1a2: set([5, 6]),
                   s2a1: set([10]), s2a2: set([10, 11])}

        eid2source2element = collections.defaultdict(bdsa_utils.dd_set_generator)
        pipeline.cluster_utils.partition_using_agglomerative(edges, eid2source2element, SaClusterRules(sa2urls))
        for aid, source2elem in eid2source2element.items():
            print("\nAID: %d, elements:"%(aid))
            for source, elems in source2elem.items():
                print("source %s - atts %s"%(source.site, ';'.join([att.name for att in elems])))


if __name__ == '__main__':
    unittest.main()
