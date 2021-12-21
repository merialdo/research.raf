import statistics
import unittest

import math

from config import constants
from utils import prob_utils

class TestProbability(unittest.TestCase):
    def test_basic(self):
        """
        See what happens changing value frequency AND number of matches
        Ignore IDF, ignore linkage accuracy (fixed at 0.9) 
        :return: 
        """
        idf = {5: 0.1, 18:1}

        #frequences = [2 ** (x) for x in range(-10,1)]
        frequences = [x/10. for x in range(1, 11)]

        value_correspondances_base = [(5,5,0.9)]
        node0 = {constants.NUMBER_OF_VALUES: 50, constants.NAME: 'aaaa'}
        node1 = {constants.NUMBER_OF_VALUES: 50, constants.NAME: 'aabb'}

        data = {}
        for i in frequences:
            for j in range(2, 10):
                node0[constants.VALUE_FREQUENCES] = {5: i, 18: 1-i}
                node1[constants.VALUE_FREQUENCES] = {5: i, 18: 1-i}
                val = prob_utils.compute_attribute_equivalent_probability(value_correspondances_base * j, node0, node1, idf)
                data[(i,j)] = val

    def test_idf(self):
        """
        See what happens changing value frequency AND number of matches
        Ignore IDF, ignore linkage accuracy (fixed at 0.9) 
        :return: 
        """

        #frequences = [2 ** (x) for x in range(-10,1)]
        frequences = [2 ** -x for x in range(1, 11)]

        value_correspondances_base = [(5,5,0.9)]
        node0 = {constants.NUMBER_OF_VALUES: 50, constants.NAME: 'aaaa'}
        node1 = {constants.NUMBER_OF_VALUES: 50, constants.NAME: 'aabb'}
        node0[constants.VALUE_FREQUENCES] = {5: 0.1, 18: 0.9}
        node1[constants.VALUE_FREQUENCES] = {5: 0.1, 15: 0.9}

        data = {}
        for i in frequences:
            for j in range(2, 10):
                idf = {5: i, 18: min(frequences), 15: 0.1}
                #TODO COPIEEEEEED !!!!!!!!!!!!!!!!!!!!!!!!!!
                avg_idf = statistics.median(idf.values())
                match_weight = {x: math.exp(-y / avg_idf) for x, y in idf.items()}

                val = prob_utils.compute_attribute_equivalent_probability(value_correspondances_base * j, node0, node1,
                                                                          match_weight)
                data[(i,j)] = val

        plot_utils.make_colormap(data)


if __name__ == '__main__':
    unittest.main()