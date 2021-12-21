import filecmp
import glob
import os
import shutil

import project_constants
from adapter import output_and_gt_adapter
from config.bdsa_config import _config_

from launcher import bdsa_launcher
from model import dataset
from scripts.results_evaluation import ResultsEvaluator, SchemaLevelEvaluation

TAG_LAUNCH = 'fx_tests_%s'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import unittest


def show_output(detail_file):
    ds = dataset.import_csv(detail_file)
    res = 'cid\tsource\tname\n'
    res += '\n'.join(','.join((row['cluster_id'], row['source'], row['name'])) for row in ds.rows)
    return res

# TODO duplicated, move to io_utils?
def get_latest_file(directory, pattern):
    all_files = glob.glob(os.path.join(directory, pattern))
    return max(all_files, key=os.path.getctime)

class FunctionalTestLauncher(unittest.TestCase):
    """
    Launches main algorithm against a test dataset and tests the results
    """

    def test_tag_naming(self):
        self._internal_test_launcher('expected.csv', 'tag-p', 3, 'instance_level_expected.csv')

    # def test_tag_naming_blacklist(self):
    #     self._internal_test_launcher('expected.csv', 'tag-b', 3, 'instance_level_expected.csv')
    #
    # #def test_mi1(self):
    # #    self._internal_test_launcher('expected_mi.csv', 'mi', 1)
    #
    # def test_mixbl(self):
    #     self._internal_test_launcher('expected_bl.csv', 'mixbl', 50, 'expected_bl_instance.csv', config_file='baseline_configuration.ini')

    def _internal_test_launcher(self, schema_level_gt_file:str, mode:str, nb=0, instance_level_gt_file=None,
                                config_file='functional_test_config.ini'):
        fx_test_config = os.path.join(THIS_DIR, config_file)
        config_target = os.path.join(project_constants.ROOT_DIR, 'config', 'algo_parameters.ini')
        shutil.copyfile(fx_test_config, config_target)
        _config_.reset_config()
        launch_tag_name = TAG_LAUNCH % mode
        bdsa_launcher.launch_bdsa(mode, 0, 0, launch_tag_name, nb)
        detail_file = get_latest_file(_config_.get_output_dir(), 'cluster_detail_%s_*.csv' % launch_tag_name)

        evaluator = ResultsEvaluator(os.path.join(THIS_DIR, schema_level_gt_file),
                                     os.path.join(THIS_DIR, instance_level_gt_file) if instance_level_gt_file else None)
        ikgpp_path = None
        if instance_level_gt_file:
            ikgpp_path = get_latest_file(_config_.get_output_dir(), 'ikgpp_%s_*.json' % launch_tag_name)
        results = evaluator.launch_evaluation_files(detail_file, ikgpp_path, schema_level_evaluation=SchemaLevelEvaluation.NON_WEIGHTED,
                                          do_partitioned_evaluation=False, output_gt_comparison=False)
        self.assertEqual((1.0, 1.0), (results.schema.precision, results.schema.recall),
                         'Output is not as expected. Output: \n%s' % show_output(detail_file))
        if instance_level_gt_file:
            self.assertEqual((1.0, 1.0), (results.instance_lib.precision, results.instance_lib.recall),
                            'Instance-level mapping is not as expected.')
            self.assertEqual((1.0, 1.0), (results.instance_cons.precision, results.instance_cons.recall),
                             'Instance-level mapping is not as expected.')
        for file in glob.glob(os.path.join(_config_.get_output_dir(), '*_%s_*' % launch_tag_name)):
            os.remove(file)

if __name__ == '__main__':
    unittest.main()


