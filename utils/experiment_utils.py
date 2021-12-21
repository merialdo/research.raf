import collections

from utils import bdsa_utils

class EvaluationMetrics:
    """
    Precision and recall metric
    """

    def __init__(self, true_positives, expected_positives, computed_positives):
        precision = -1 if computed_positives == 0 else true_positives / float(computed_positives)
        recall = -1 if expected_positives == 0 else true_positives / float(expected_positives)
        f_measure = 0 if precision + recall == 0 else \
            2 * precision * recall / float(precision + recall)

        self.precision = bdsa_utils.round_sig(precision, 3)
        self.recall = bdsa_utils.round_sig(recall, 3)
        self.f_measure = bdsa_utils.round_sig(f_measure, 3)
        self.true_positives = true_positives
        self.expected_positives = expected_positives
        self.computed_positives = computed_positives
        self.subsets = {}

    def _asdict(self):
        return {'precision': self.precision, 'recall': self.recall, 'F-Measure': self.f_measure}

    def __str__(self):
        res = 'P: %.2f, R: %.2f, F1: %.2f' % (self.precision, self.recall, self.f_measure)
        res += '\nTP:%d, FP: %d, FN: %d' % (self.true_positives, self.computed_positives-self.true_positives,
                                            self.expected_positives - self.true_positives)
        if len(self.subsets) > 0:
            res += '\nAlso following subsets available: %s' % ','.join(self.subsets.keys())
        return res

    def pr(self):
        return 'P: %f, R: %f' % (self.precision, self.recall)

    def __repr__(self):
        return self.__str__()

    def compact(self):
        return (self.true_positives, self.expected_positives, self.computed_positives)

    def add_subset(self, name, true_positives, expected_positives, computed_positives):
        """
        Add evaluation on portions of dataset
        :param name:
        :param true_positives:
        :param expected_positives:
        :param computed_positives:
        :return:
        """
        self.subsets[name] = EvaluationMetrics(true_positives, expected_positives, computed_positives)

    def add_subset(self, name, subeval):
        """
        Add evaluation on portions of dataset
        :param name:
        :param subeval: sub-evaluation eval metrics
        :return:
        """
        self.subsets[name] = subeval

def evaluation_metrics_with_falses(tp, fp, fn):
    """
    Factory for evaluation metrics that uses false positives and false negatives
    :param tp:
    :param fp:
    :param fn:
    :return:
    """
    return EvaluationMetrics(tp, tp + fn, tp + fp)

def evaluate_dataset(computed_clusters: list, expected_clusters: list, category2function={}) -> (
EvaluationMetrics, dict):
    """
    Evaluate precision, recall and F-Measure, given real and computed clustering.

    The evaluation is based on the following model:
    - INPUT: all possible pair of attributes
    - OUTPUT: 1-0, if they are or not in the same cluster. 

    Also detect metrics for specific categories of nodes (e.g. if nodes are numbers, categories may
    be numbers > 100 or nb < 10).

    Note that some clusters may contain elements of different categories: in this case, each pair with mixed
    category is counted as 0.5 in that category.    

    :param computed_clusters: 
    :param expected_clusters: 
    :param category2function:
    :return: 
    """

    category2computed_positives, category2expected_positives, category2true_positives, computed_positives_global, \
    expected_positives_global, true_positives_global = evaluate_expected_computed(
        category2function, computed_clusters, expected_clusters)
    global_metrics = EvaluationMetrics(true_positives_global, expected_positives_global, computed_positives_global)
    category_metrics = {cat: EvaluationMetrics(
        category2true_positives[cat], category2expected_positives[cat], category2computed_positives[cat])
        for cat in category2function.keys()}
    return global_metrics, category_metrics


def evaluate_expected_computed(category2function, computed_clusters, expected_clusters):
    """
    See evaluate_dataset for details. This method does the same but does not compute P-R, it returns the nb of true 
    positives, computed positives and expected positives. Useful also to compute differences between 2 algorithms
    :param category2function: 
    :param computed_clusters: 
    :param expected_clusters: 
    :return: 
    """
    # Element to ID of a cluster it is expected to belong
    inverted_expected_clusters = {}
    cluster_id = 0
    # Expected and computed positives, globally and for each specific category
    expected_positives_global = 0
    computed_positives_global = 0
    true_positives_global = 0
    category2expected_positives = collections.Counter()
    category2computed_positives = collections.Counter()
    category2true_positives = collections.Counter()
    # Expected clusters may be a subset (golden set) of data. So keep only elements in computed that are also in expected
    elements_in_golden_set = set()
    # Expected clusters: compute total expected, and build an inverted index (used later for true positives)
    for cluster in expected_clusters:
        expected_positives_global += \
            increment_global_and_per_category_positives(category2expected_positives, category2function, cluster)
        inverted_expected_clusters.update({elem: cluster_id for elem in cluster})
        elements_in_golden_set.update(cluster)
        cluster_id += 1
    for comp_cluster in computed_clusters:
        cluster_filtered = [elem for elem in comp_cluster if elem in elements_in_golden_set]
        computed_positives_global += \
            increment_global_and_per_category_positives(category2computed_positives, category2function,
                                                        cluster_filtered)
        # To compute true positives, we now have to partition the cluster in chunks
        # according to the expected clusters id.
        # Ideally there would be only 1 partition because all elements are from a same expected cluster
        # Example: say expected clusters are [1,1,1,1] and [2,2,2,2]
        # If computed cluster is [1,1,1,2,2] then we have 2 chunks: [1,1,1] and [2,2], that makes 3+1=4 TP

        expected_cluster_id2elements = collections.defaultdict(list)
        for elem in cluster_filtered:
            expected_cluster_id2elements[inverted_expected_clusters[elem]].append(elem)
        for chunk in expected_cluster_id2elements.values():
            true_positives_global += increment_global_and_per_category_positives(category2true_positives,
                                                                                 category2function,
                                                                                 chunk)
    return category2computed_positives, category2expected_positives, category2true_positives, computed_positives_global, expected_positives_global, true_positives_global


def increment_global_and_per_category_positives(category2nb_positives, category2function, cluster):
    """
    Increment nb of expected or computed positives in a cluster, globally and according to categories

    :param category2nb_positives: dict to increment 
    :param category2function: 
    :param cluster: 
    :param positives_global: number to increment
    :return: 
    """
    category2nb_positives += {cat: compute_category_combinations_in_cluster(cluster, fx)
                              for cat, fx in category2function.items()}
    return compute_nb_combinations(len(cluster))


def compute_category_combinations_in_cluster(cluster, fx):
    """
    Provided a cluster and a function determining a category, computes the count of pairs of that category in the cluster
    This number of pair can be used for either expected positives or computed positives, if cluster is
    expected or generated.
    Example: cluster is [1,2,3,4, 101,102], category is [> 100]. Then we have:
    - 1 full-expected positive pair (101-102)
    - 8 mid-expected pos pair, 1-101, 1-102... that are hybrid pairs so will be counted 0.5
    - So result = 1+8/2 = 5

    :param cluster: 
    :param fx: 
    :return: 
    """
    nb_category_element = sum(1 for elem in cluster if fx(elem))
    nb_noncategory_element = len(cluster) - nb_category_element
    hybrid_pairs = nb_category_element * nb_noncategory_element
    full_pairs = compute_nb_combinations(nb_category_element)
    return 0.5 * hybrid_pairs + full_pairs


def compute_nb_combinations(size_cluster):
    """
    Provided a collection size, compute nb of possible 2-combinations
    :param size_cluster: 
    :return: 
    """
    return size_cluster * (size_cluster - 1) / float(2)
