import unittest
from utils import experiment_utils

class TestEvaluations(unittest.TestCase):
    def test_basic(self):
        """
        First is computed, then expected. 
        - TP: 2-3, 103-104 [2]
        - CP: 2-3-4*, 6-102, 103-104 [5]
        - EP: 1-2-3*, 4-5-6*, 102-103-104* [9]
        computed:
        :return: 
        """
        glob, categ = experiment_utils.evaluate_dataset([[1], [2,3,4], [5], [6,102], [103, 104]],
        [[1, 2, 3], [4, 5, 6], [102, 103, 104]], {'>100': lambda x: x > 100, '<100': lambda x: x < 100})
        self.assertAlmostEqual(2/5., glob.precision, delta=0.01)
        self.assertAlmostEqual(2 / 9., glob.recall, delta=0.01)
        self.assertAlmostEqual(1 / 3.5, categ['<100'].precision, delta=0.01)
        self.assertAlmostEqual(1 / 6., categ['<100'].recall, delta=0.01)
        self.assertAlmostEqual(1 / 1.5, categ['>100'].precision, delta=0.01)
        self.assertAlmostEqual(1 / 3., categ['>100'].recall, delta=0.01)

if __name__ == '__main__':
    unittest.main()