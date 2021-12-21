import math

import numpy as np
import itertools
import random
import collections

### Compute statistics over data
from tqdm import tqdm

HEAD = 'H'
TAIL = 'T'
HT = 'head/tail'


class NamedLambda():
    def __init__(self, func, name, cachable=False):
        self._func = func
        self._name = name
        self._cachable = cachable
        if cachable:
            self._cache = {}

    def apply(self, x):
        if not self._cachable:
            return self._func(x)
        else:
            if x not in self._cache:
                self._cache[x] = self._func(x)
            return self._cache[x]

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name


class GroupSize():
    """
    This class is useful to compute stats on group of elements.
    For instance clusters, let's say we have clusters of 10,5,3,1,1
    --> avg=4; median=3; HT: 1 head avg 10; 4 tail avg 2.5
    --> plot
    """

    def __init__(self, element_sizes):
        head_tail = collections.defaultdict(list)
        compute_head_tail(element_sizes, lambda x: x, lambda element, ht: head_tail[ht].append(element))

        self.size = len(element_sizes)
        self.avg = np.mean(element_sizes)
        self.median = np.median(element_sizes)

        self.head_size = len(head_tail[HEAD])
        self.head_avg = np.mean(head_tail[HEAD])
        self.head_median = np.median(head_tail[HEAD])

        self.tail_size = len(head_tail[TAIL])
        self.tail_avg = np.mean(head_tail[TAIL])
        self.tail_median = np.median(head_tail[TAIL])

    def __str__(self):
        gen = "***GENERAL***: Size %d, Average: %f, median: %f" % (self.size, self.avg, self.median)
        head = "***HEAD***: Size %d, Average: %f, median: %f" % (self.head_size, self.head_avg, self.head_median)
        tail = "***TAIL***: Size %d, Average: %f, median: %f" % (self.tail_size, self.tail_avg, self.tail_median)
        return '\n'.join((gen, head, tail))


def compute_head_tail_dataset(the_dataset, size_key):
    """
    Group elements in head and tail according to the size_key
    :param the_dataset: 
    :param size_key: 
    :return: 
    """
    # this is necessary as we do not add new rows with add_row, but rather edit existing ones
    the_dataset.add_element_to_header(HT)
    compute_head_tail(the_dataset.rows, lambda element: int(element[size_key]),
                      lambda element, ht: assign_value(element, HT, ht))


def assign_value(ditto, key, value):
    """
    Just assign a value to a key in a dictionary.
    Workaround for lambda functions, as [lambda x: dicto[x] = y] does not work
    :param ditto: 
    :param key: 
    :param value: 
    :return: 
    """
    ##TODO is there a better solution?
    ditto[key] = value


def compute_head_tail(all_elements, get_size_key, add_head_tail):
    """
    Group elements in head and tail according to the size_key.
    Generic method
    :param all_elements: 
    :param get_size_key: 
    :param add_head_tail:
    :param external_tail_elements_size: if you have tail elements already analyzed and not present in all_elements then put here their total size
    :return: 
    """
    median = sum(get_size_key(element) for element in all_elements) / float(2)
    sorted_elements = sorted(all_elements, key=get_size_key, reverse=True)
    cumulate = 0
    # We want to group element by size, that is because we want to keep ALL elements of a certain size either in head or in tail
    for size_key, elements in itertools.groupby(sorted_elements, key=get_size_key):
        if cumulate < median:
            ht = HEAD
        else:
            ht = TAIL
        for element in elements:
            add_head_tail(element, ht)
            cumulate += size_key


def compute_entropy(values, bucket_step):
    """

    :param values: a list of numerical values
    :param bucket_step:
    :return:
    """
    number_of_elements = float(len(values))
    buckets = list(_compute_buckets(values, bucket_step).values())
    ratios = [bucket / number_of_elements for bucket in buckets]
    return sum([-k * math.log(k) for k in ratios])


def _compute_buckets(values, bucket_step):
    """
    Given a bucket step,
    :param values:
    :param bucket_step:
    :return:
    """
    results = collections.defaultdict(int)
    for element in values:
        the_bucket = int(element / bucket_step)
        results[the_bucket] += 1
    return results


def sample_dataset(granularity, size_sample, elements, continous_functions, categorical_functions,
                   print_repartition_matrix=False):
    """
    Builds a sample dataset of the provided elements.
    Some measures should be provided, the method will return a dataset in which the distribution of these measures
    is similar to that of original dataset.

    E.g.: if a measure is 'length', and in original dataset there are 30 elements with 0 < length < 5; 300 with
    5 < length < 10 and 70 with length > 10, and we want 40 sample elements, then the dataset will contain approximately
    3,30,7 elements of the respective intervals.

    :param granularity: the number of intervals that should be computed for each measure
    :param size_sample: the expected cardinality of the sample (the result may be slightly different)
    :param elements: the original dataset
    :param continous_functions: the list of measures (that return a continous number as output)
    :param categorical_functions: the list of measures (that return categorical output, e.g. a category)
    :return:
    """
    if granularity <= 1:
        raise Exception("Granularity must be 2 at least")

    if len(elements) <= 1:
        print("%d elements, returning all of them..." % len(elements))
        return elements

    function2controls = {}

    # compute controls
    for sample_function in continous_functions:
        values = [sample_function.apply(elem) for elem in elements]
        min_value = min(values)
        max_value = max(values)
        step = (max_value - min_value) / float(granularity)
        if step == 0:
            print('Values are all the same for [%s], no repartition will be done' % (str(sample_function)))
        else:
            the_range = np.arange(min_value + step, max_value, step)
            function2controls[sample_function] = _build_interval_control(the_range)

    # build controls for CATEGORICAL functions
    for cat_function in categorical_functions:
        values = set(cat_function.apply(elem) for elem in elements)
        control_functions = [NamedLambda(equality_lambda(value), "x = %s" % (str(value))) for value in values]
        function2controls[cat_function] = control_functions

    print("Controls computed, building repartition matrix (this can be long...)")
    matrix = _build_repartition_matrix(elements, function2controls)

    if print_repartition_matrix:
        print({x: len(y) for x, y in matrix.items()})

    # how much we want to reduce the dataset
    reduction = max(1, len(elements) / size_sample)
    sample = []
    print("Building sample...")
    for elements in list(matrix.values()):
        sample.extend(random.sample(elements, max(1, round(len(elements) / reduction))))
    return sample


def equality_lambda(value):
    """
    Returns an equality function.
    We need to use an external function otherwise it does not work
    :param value:
    :return:
    """
    return lambda x: x == value


def count_elements(elements, get_value=lambda x: x):
    counter = collections.defaultdict(int)
    for elem in elements:
        value = get_value(elem)
        counter[value] += 1
    return counter


def _build_repartition_matrix(elements, functions2controls):
    """
    Distribute a list of elements in a matrix according to some measures.
    For instance: we have a list of cellphones and we want to distribute them by length and width.

                   0 < len < 15cm    15< len < 20         len > 20
                ---------------------------------------------------------
    width<5     | KL55; L545    |  iphone6      |     ASUSZE550         |
    5<width<10  | 546565        |  iphone7      |         //            |
    width>10    |  KL741        |  llk,565      |     kh42,45           |
                ---------------------------------------------------------


    :param elements: list of elements to distribute
    :param functions2controls: for each axis, a function (how to compute the measure) and a list of controls
    (does the measure fits a specific range)
    E.g: {(lambda elem: elem.length) : [lambda length: length < 15, lambda length: length < 15 and length < 20....), ...}
    :return:
    """

    # example: if we have length --> 0..1, 1..2, 2..3 and width: 2..3 and 3..4
    # this will return (length, 0..1, width, 2..3) ; (length, 0..1, width, 3..4); ....
    # each tuple corresponds to a case of the matrix
    all_controls = [[NamedLambda(_combine_fx(control.apply, fx.apply), str(fx) + "__" + str(control))
                     for control in controls] for fx, controls in functions2controls.items()]
    product = list(itertools.product(*all_controls))
    values_matrix = collections.defaultdict(list)

    for list_of_controls in tqdm(product): #Weight over nb of elements?
        for element in elements:
            # verify if element goes in the cell according to list_of_controls
            elements_goes_in_cell = True
            for fx_control in list_of_controls:
                elements_goes_in_cell = elements_goes_in_cell and fx_control.apply(element)
            if elements_goes_in_cell:
                values_matrix[str(list_of_controls)].append(element)
    return values_matrix


def _build_interval_control(the_range):
    """
    Builds a list of controls over a set of ranges

    :param the_range: list of elements, e.g. 1,2,3
    :return: list of functions, e.g [x<1, 1<=x<2; 2<=x<3; x>3]
    """
    list_of_controls = []
    list_of_controls.append(NamedLambda(_interval_fx(None, the_range[0]), "x < %f" % (the_range[0])))
    for before, after in zip(the_range, the_range[1:]):
        list_of_controls.append(NamedLambda(_interval_fx(before, after), "%f <= x < %f" % (before, after)))
    list_of_controls.append(NamedLambda(_interval_fx(the_range[-1], None), "x >= %f" % (the_range[-1])))
    return list_of_controls


def _combine_fx(fx1, fx2):
    return lambda x: fx1(fx2(x))


def _interval_fx(lower, upper):
    if lower and upper:
        return lambda x: x >= lower and x < upper
    elif lower:
        return lambda x: x >= lower
    else:
        return lambda x: x < upper


def safe_divide(number1, number2):
    if number2 == 0:
        return 0
    else:
        return float(number1) / number2

def safe_sample(elements, k):
    """
    Like random.sample but does not throw an error if k > len(elements)
    """
    return random.sample(elements, min(k, len(elements)))

def jaccard_similarity(set1:set, set2:set):
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)