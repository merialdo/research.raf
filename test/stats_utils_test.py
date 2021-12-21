import unittest

from utils import stats_utils

_debug_ = True

class TestStatsUtils(unittest.TestCase):
    def test_evaluation(self):

        #1 split
        self._aux_method_evaluation([[0,1,2,3,4,5]], [[0,1,2],[3,4,5]], 6/float(15), 1 )
        self._aux_method_evaluation([[0,1,2],[3,4,5]], [[0,1,2,3,4,5]], 1, 6/float(15))
        #Try with different orders
        self._aux_method_evaluation([[3,4,5], [2,0,1]], [[0,1,2,3,4,5]], 1, 6/float(15))


        #Different repartition
        self._aux_method_evaluation([[0,1,2],[3]], [[0,1],[2,3]], 1 / float(3), 1/float(2))

        #If no detected or no real, result is zero (actually there is a divide by zero)
        self._aux_method_evaluation([[0],[1],[2]], [[0,1,2]], 0, 0)
        self._aux_method_evaluation([[0,1,2]], [[0],[1],[2]], 0, 0)
        self._aux_method_evaluation([], [], 0, 0)
        self._aux_method_evaluation([range(1000)], [range(500), range(500,1000)], 499. / 999, 1)

    def _aux_method_evaluation(self, computed_data, real_data, exp_precision, exp_recall):
        p,r,f = stats_utils.evaluate_dataset(computed_data, real_data)
        if _debug_:
            print("Real: %s, computed: %s, P %f, r %r, f %f"%(str(real_data), str(computed_data), p,r,f))
        self.assertAlmostEqual(p, exp_precision)
        self.assertAlmostEqual(r, exp_recall)

if __name__ == '__main__':
    unittest.main()