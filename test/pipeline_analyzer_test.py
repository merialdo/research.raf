import unittest

from config import constants
from model import datamodel
from pipeline import pipeline_analyzer
from pipeline.pipeline_common import ClusteringOutput
from model.bdsa_data import BdsaData
from test.mock_linkage_adapter import MockLinkageAdapter
from config.bdsa_config import _config_

SITE_EXAMPLE = 'www.site%d--s--H--s--10--s--100--s--700.com'
ATT_EXAMPLE = 'att%d@@@H@2@0,100000@15-0'

#FIXME set new evaluation elements
class TestPipelineAnalyzer(unittest.TestCase):
    def test_evaluation(self):
        _config_.config.set('parameters', 'do_synthetic_evaluation', 'yes')
        sa_clusters = {
            1: {
                source(1): [att(1,1), att(1,2)]
            },
            2: {
                source(1): [att(1,3),att(1,4)],
                source(2): [att(2,1),att(2,2)],
                source(3): [att(3,3),att(3,4)]
            }
        }
        sa_isolated = {
            att(1, 4):{}, att(1,5): {}
        }

        page_clusters = {
            1: {
                source(2):[page(2,1),page(2,2)]
            },
            2: {
                source(1): [page(1,2),page(1,7)],
                source(3): [page(3,2),page(3,3)],
                source(2): [page(4,3),page(4,4)]
            }
        }
        page_isolated = {
            page(1, 7):{}, page(1,8): {}
        }

        sa2size = {
            att(1, 4): 1, att(1, 5): 2,
            att(5,6): 7
        }

        results = ClusteringOutput() #sa_clusters, sa_isolated, None, sa2size, page_clusters, page_isolated
        results.sa_clusters = sa_clusters
        results.sa_isolated = sa_isolated
        results.bdsa_data = BdsaData()
        results.bdsa_data.sa2size = sa2size
        results.page_clusters = page_clusters
        results.page_isolated = page_isolated
        _config_.get_output_dir()
        analyzer = pipeline_analyzer.PipelineAnalyzer('tag')
        stats = analyzer.compute_stats_on_graph(results)
        self.assertEqual(0.8, stats[constants.STATS_PERC_ATTRIBUTES_NON_ISOLATED])
        self.assertEqual(10, stats[constants.STATS_NUMBER_ATTRIBUTES])
        self.assertEqual(4, stats[constants.STATS_AVG_CLUSTER_SIZE])
        self.assertEqual(8, stats[constants.STATS_NUMBER_PAGES_NON_ISOLATED])
        self.assertEqual(2, stats[constants.STATS_NUMBER_ISOLATED_PAGES])
        self.assertEqual(2, stats[constants.STATS_NUMBER_PAGE_CLUSTERS])
        self.assertEqual(0.7, stats[constants.STATS_PERC_VALUES_NONISOLATED])
        analyzer.compute_evaluation(stats, results)
        sa_measures = stats[pipeline_analyzer.SA_MEASURES][pipeline_analyzer.GLOBAL]
        self.assertAlmostEqual(0.125, sa_measures['P'], delta=0.01)
        self.assertAlmostEqual(0.33, sa_measures['R'], delta=0.01)
        page_measures = stats[pipeline_analyzer.PAGE_MEASURES][pipeline_analyzer.GLOBAL]
        self.assertAlmostEqual(0.125, page_measures['P'], delta=0.01)
        self.assertAlmostEqual(0.4, page_measures['R'], delta=0.01)
        pass

def att(site, name):
    return datamodel.source_attribute_factory('cat', SITE_EXAMPLE % site, ATT_EXAMPLE % name)

def page(site, url):
    return datamodel.page_factory('%s/%d/'%(SITE_EXAMPLE % site, url), source(site))

def source(sid):
    return datamodel.SourceSpecifications(SITE_EXAMPLE % sid, 'cat', None)

if __name__ == '__main__':
    unittest.main()