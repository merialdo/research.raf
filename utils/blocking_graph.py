import collections
import itertools

from utils import bdsa_utils


class MetablockingGraph:
    """
    Use this class to select candidate entity pairs for linkage/alignment.
    It will build weighted edges between entities, and select pairs with a weight bigger than a threshold.
    Note that elements are grouped into pair of 'groups' (i.e. a category in which each entity belongs)
    """

    def __init__(self, threshold):
        self.candidates = collections.defaultdict(bdsa_utils.dd_float_generator)
        self.threshold = threshold
        pass

    def add_full_clique(self, source2entity_list:list, score:int):
        """
        For each set of entities provided (grouped by source), add edges between ALL pair of entities of different source,
        with given score
        :param source2entity:
        :return:
        >>> mb = MetablockingGraph(2)
        >>> mb.add_full_clique([{'s1': {1,2}, 's2': {3,4}}, {'s3': {5}, 's2': {1}}], 2)
        >>> mb.get_candidates()
        {('s1', 's2'): {(1, 3): 2.0, (1, 4): 2.0, (2, 3): 2.0, (2, 4): 2.0}, ('s2', 's3'): {(1, 5): 2.0}}
        """

        for source2entity in source2entity_list:
            for source1, source2 in itertools.combinations(source2entity.keys(), 2):
                for page1, page2 in itertools.product(source2entity[source1], source2entity[source2]):
                    self.increment_weigth(page1, page2, source1, source2, score)

    def add_single_clique_flat(self, entity_list, group_getter, score):
        entity_sorted = sorted(entity_list, key=group_getter)
        source2entities = {}
        for source, entities in itertools.groupby(entity_sorted, key=group_getter):
            source2entities[source] = list(entities)
        self.add_full_clique([source2entities], score)


    def increment_weigth(self, entity1, entity2, group1, group2, weight):
        if group1 < group2:
            group_pair = (group1, group2)
            entity_pair = (entity1, entity2)
        else:
            group_pair = (group2, group1)
            entity_pair = (entity2, entity1)
        self.candidates[group_pair][entity_pair] += weight

    def get_candidates(self):
        """
        Call this method to get all valid candidates
        >>> bg = MetablockingGraph(4)
        >>> bg.increment_weigth(15,22,1,2, 2)
        >>> bg.increment_weigth(15,27,1,2, 1)
        >>> bg.increment_weigth(22,15,2,1, 2)
        >>> bg.increment_weigth(15,27,1,2, 2)
        >>> bg.increment_weigth(15,35,1,3, 4)
        >>> bg.increment_weigth(15,47,1,4, 3)
        >>> bg.get_candidates()
        {(1, 2): {(15, 22): 4.0}, (1, 3): {(15, 35): 4.0}}
        
        :return: 
        """
        group_pair2entity_pair2score = {}

        for group_pair, entity_pairs in self.candidates.items():
            res = {page_pair:score for page_pair, score in entity_pairs.items() if score >= self.threshold}
            if len(res) > 0:
                group_pair2entity_pair2score[group_pair] = res
        return group_pair2entity_pair2score