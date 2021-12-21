import collections
import itertools

from adapter import adapter_factory
from adapter.abstract_linkage_adapter import AbstractLinkageAdapter
from model import datamodel
from model.datamodel import SourceSpecifications, SourceAttribute
from model.dataset import Dataset
from sandbox import launch_scripts
from utils import io_utils, string_utils

from config.bdsa_config import _config_

#PREDICTION_FILE = 'C:/Users/fpiai/proj/bdsa/LinkPrediction/predictions/camera_dataset/alpha-100'
PREDICTION_FILE = 'C:/Users/fpiai/proj/bdsa/LinkPrediction/predictions/camera_dataset_instance/alpha-100'

SPACE_SEPARATOR = '--'

SOURCE_ATT_SEPARATOR = '___'

INSTANCE_OF = 'instanceOf'

DUMMY = 'dummy'

UNKNOWN = 'UNKNOWN'


class Triple:
    """
    A fact, composed of an entity ID, a source attribute AND a value
    """

    def __init__(self, entity_id:str, sa: SourceAttribute, value:str):
        self.entity_id = entity_id
        self.sa = sa
        self.value = value

    def __str__(self):
        return ' - '.join([self.entity_id, str(self.sa), self.value])

def triple_factory(entity_id:str, ss: SourceSpecifications, key:str, value:str):
    return Triple(entity_id, datamodel.source_attribute_factory(ss.category, ss.site, key), value)

def pt(eid, site, name, value, heads, tails):
    """
    Prediction builder for tests
    :return:
    """
    sa = datamodel.source_attribute_factory('dummy', site, name)
    triple = Triple(eid, sa, value)
    return Prediction(triple, heads, tails)

class Prediction:
    def __init__(self, triple, heads, tails):
        self.triple = triple
        self.heads = heads
        self.tails = tails

    def __repr__(self):
        return '%s\n%s ; %s' % (str(self.triple), str(self.heads), str(self.tails))

# TODO take all combinations of entities + relations with name=brand
# if exists, put, otherwise put UNKNOWN



def _convert_triple_to_tsv_row(triple):
    """
    Convert to a row of link prediction algorithms (anyBURL and rotatE).
    :return:
    """
    eid_mod = triple.entity_id.replace(' ','--')
    name_mod = triple.sa.name.replace(' ', '--')
    value_mod = triple.value.replace(' ', '--')
    return '%s\t%s%s%s\t%s' % (eid_mod, triple.sa.source.site, SOURCE_ATT_SEPARATOR, name_mod, value_mod)

def convert_bdsa_to_anyburl():
    """
    Convert data from BDSA to anyburl train file
    :return:
    """
    # convert to intermediate triple list
    train_triples = list(_import_bdsa_data_as_triples(_source_to_train_triples))
    test_triples = list(_import_bdsa_data_as_triples(_source_to_test_triples))
    _build_output(test_triples, train_triples)

def convert_bdsa_to_anyburl_instances():
    """
    Convert data from BDSA to anyburl train file
    :return:
    """
    # convert to intermediate triple list
    train_triples = list(_import_bdsa_data_as_triples(_source_to_train_triples_instance))
    test_triples = list(_import_bdsa_data_as_triples(_source_to_test_triples_instance))
    _build_output(test_triples, train_triples)


def _build_output(test_triples, train_triples):
    # convert triple lists to file
    path = io_utils.build_directory_output('link_prediction_data')
    io_utils.output_file_generic(train_triples, _convert_triple_to_tsv_row, 'train', directory=path, timestamp=False)
    io_utils.output_file_generic(test_triples, _convert_triple_to_tsv_row, 'test', directory=path, timestamp=False)
    io_utils.output_file_generic(test_triples, _convert_triple_to_tsv_row, 'valid', directory=path, timestamp=False)


def _import_bdsa_data_as_triples(bdsa_source_to_triple_converter):
    """
    Import BDSA and convert to triple
    :return:
    """
    sgen = adapter_factory.spec_factory().specifications_generator()
    linkage = adapter_factory.linkage_factory()
    for source in sgen:
        yield from bdsa_source_to_triple_converter(linkage, source)

def _source_to_train_triples(linkage, source):
    """
    Given a bdsa source, converts all to triples (as training data)
    :param linkage:
    :param source:
    :param specs:
    :param url:
    :return:
    """
    for url, specs in source.pages.items():
        ids = linkage.ids_by_url(url, source.site, source.category)
        for entity_id, keyvalue in itertools.product(ids, specs.items()):
            if keyvalue[0] not in _config_.get_excluded_attribute_names():
                yield triple_factory(entity_id, source, keyvalue[0], keyvalue[1])


def _source_to_train_triples_instance(linkage, source):
    """
    Same as _source_to_train_triples but triples are duplicated: url instanceof pid + pid attr value
    :param linkage:
    :param source:
    :return:
    """
    for url, specs in source.pages.items():
        ids = linkage.ids_by_url(url, source.site, source.category)
        for pid in ids:
            yield Triple(pid, datamodel.source_attribute_factory(DUMMY, DUMMY, INSTANCE_OF), url)
        for key, value in specs.items():
            yield triple_factory(url, source, key, value)

def _source_to_test_triples_instance(linkage, source):
    """
    Same as _source_to_test_triples but triples are duplicated: url instanceof pid + pid attr value
    :param linkage:
    :param source:
    :return:
    """
    sas_involved = set()

    # First, retrieve all brand* and *manufacturer sas and all PIDS, so we can make the product.
    # Also associate PID with SPECS so we can know the actual relations
    for url, specs in source.pages.items():
        sa_filtered = [datamodel.source_attribute_factory(source.category, source.site,key) for key in specs.keys()
                          if key.lower() in ['brand', 'manufacturer', 'brand name', 'manufacturer name']]
        sas_involved.update(sa_filtered)
    # Then, do the product
    for url, sa in itertools.product(source.pages.keys(), sas_involved):
        value = source.pages[url].get(sa.name, UNKNOWN)
        yield Triple(url, sa, value)

def _source_to_test_triples(linkage: AbstractLinkageAdapter, source):
    """
    Convert BDSA input to test train data:
    * for each source, take all brand* or *manufacturer relations
    * take the union of all IDS
    * build a file with all triples

    :param linkage:
    :param source:
    :param specs:
    :param url:
    :return:
    """
    pid2specs = collections.defaultdict(dict)
    sas_involved = set()

    # First, retrieve all brand* and *manufacturer sas and all PIDS, so we can make the product.
    # Also associate PID with SPECS so we can know the actual relations
    for url, specs in source.pages.items():
        pids = linkage.ids_by_url(url, source.site, source.category)
        specs_filtered = {datamodel.source_attribute_factory(source.category, source.site,key) :value for key, value in specs.items()
                          if key.lower() in ['brand','manufacturer','brand name','manufacturer name']}
        sas_involved.update(specs_filtered.keys())
        for pid in pids:
            pid2specs[pid].update(specs_filtered)
    # Then, do the product
    for pid, sa in itertools.product(pid2specs.keys(), sas_involved):
        value = pid2specs[pid].get(sa, UNKNOWN)
        yield Triple(pid, sa, value)

### PREDICTIONS ###

def evaluate_predictions(predictions_importer):
    """
    Evaluate how many predictions were correct
    :param predictions_importer:
    :return:
    >>> importer = lambda : [pt('iphone6', 'amz','brand', 'apple', [], [('yo',0.5), ('samsung', 0.3)]),\
    pt('iphone6', 'ebay','brand', 'apple', [], [('apple',0.5), ('yo', 0.1)]), \
    pt('mi2', 'ebay','brand', 'xiaomi', [], [('xiaomi',0.5), ('yo', 0.1)]), \
    pt('gs7', 'amz','brand', 'samsung', [], [('samsung',0.5), ('yo', 0.3)]),  \
    pt('gs7', 'ebay','brand', 'apple', [], [('apple',0.5), ('samsung', 0.1)]), \
    pt('xperia', 'ebay','brand', 'sony', [], [('sony',0.5), ('samsung', 0.1)]), \
    pt('iphone6', 'buy.net', 'brand', 'UNKNOWN', [], [('apple',0.5), ('samsung', 0.1)]),  \
    pt('iphone6', 'price-hunt.net', 'manufacturer', 'UNKNOWN', [], [('samsung',0.5), ('yo', 0.1)]),  \
    pt('power', 'amz', 'brand', 'UNKNOWN', [], [('samsung',0.5), ('yo', 0.1)]),  \
    pt('gs7', 'price-hunt.net', 'brand', 'UNKNOWN', [], [('samsung',0.5), ('yo', 0.1)]), \
    pt('gs7', 'buy.net', 'brand', 'UNKNOWN', [], [('apple',0.5), ('yo', 0.1)]),  \
    pt('xperia', 'buy.net', 'brand', 'UNKNOWN', [], [('sony',0.5), ('yo', 0.1)]), \
    pt('xperia', 'price-hunt.net', 'brand', 'UNKNOWN', [], [('sony',0.5), ('yo', 0.1)])]
    >>> evaluate_predictions(importer)
    RESULTS: 5/6 for known, 3/4 for unknown and 2 ambiguous

    """
    total_known = 0
    total_correct_known = 0
    total_correct_unknown = 0
    total_ambiguous_unknown = 0
    total_inductable_unknown = 0
    all_unknown = []
    facts_for_entity = {}
    doubtful_entities = set() # entities for which there are different values
    for pred in predictions_importer():
        if pred.triple.value != UNKNOWN:
            total_known += 1
            if len(pred.tails) > 0 and string_utils.folding_using_regex(pred.triple.value) == string_utils.folding_using_regex(pred.tails[0][0]):
                total_correct_known += 1
            entity_id = pred.triple.entity_id
            if entity_id not in doubtful_entities:
                if entity_id not in facts_for_entity:
                    facts_for_entity[entity_id] = pred.triple.value
                else:
                    if facts_for_entity[entity_id] != pred.triple.value:
                        del facts_for_entity[entity_id]
                        doubtful_entities.add(entity_id)
        else:
            all_unknown.append(pred)
    for pred in all_unknown:
        if pred.triple.entity_id in doubtful_entities:
            total_ambiguous_unknown += 1
        else:
            if pred.triple.entity_id in facts_for_entity:
                total_inductable_unknown += 1
                if len(pred.tails) > 0 and pred.tails[0][0] == facts_for_entity.get(pred.triple.entity_id, None):
                    total_correct_unknown += 1
    print ("RESULTS: %d/%d for known, %d/%d for unknown and %d ambiguous" % (total_correct_known, total_known, total_correct_unknown, total_inductable_unknown, total_ambiguous_unknown))


def import_predictions_from_file():
    """
    Import a prediction file
    :return:
    """
    predictions_file = PREDICTION_FILE  #launch_scripts.get_file_name()
    line_stream = io_utils.import_generic_file_per_line(predictions_file)
    next_pred = import_single_prediction(line_stream)
    while next_pred:
        yield next_pred
        next_pred = import_single_prediction(line_stream)

def import_single_prediction(line_stream):
    """
    Import a single triple of predictions
    :param line_stream:
    :return:
    >>> ls = iter(["http:///www.ebay.com/42149 www.ebay.com___brand canon", \
"Heads: http:///www.ebay.com/47528\\t0.9464285714285714\\thttp:///www.ebay.com/58737\\t0.6818181818181818	\\thttp:///www.ebay.com/54040\\t0.6818181818181818\\n",\
"Tails: canon\\t0.21791044776119403	\\tnikon\\t0.16616915422885573\\tsony\\t0.10845771144278607\\n"])
    >>> import_single_prediction(ls)
    http:///www.ebay.com/42149 - www.ebay.com__dummy/brand - canon
    [('http:///www.ebay.com/47528', 0.9464285714285714), ('http:///www.ebay.com/58737', 0.6818181818181818), ('http:///www.ebay.com/54040', 0.6818181818181818)] ; [('canon', 0.21791044776119403), ('nikon', 0.16616915422885573), ('sony', 0.10845771144278607)]

    """
    triple_line = _next_no_cr(line_stream, None)
    if not triple_line:
        return None

    triple_split = triple_line.split(' ')
    sa_split = triple_split[1].split(SOURCE_ATT_SEPARATOR)
    sa = datamodel.source_attribute_factory('dummy', sa_split[0], sa_split[1].replace(SPACE_SEPARATOR, ' '))
    triple = Triple(triple_split[0], sa, triple_split[2].replace(SPACE_SEPARATOR, ' '))
    heads = _retrieve_heads_or_tails(line_stream, True)
    tails = _retrieve_heads_or_tails(line_stream, False)
    return Prediction(triple, heads, tails)


def _retrieve_heads_or_tails(line_stream, is_head):
    initial_text = 'Heads: ' if is_head else 'Tails: '

    line = _next_no_cr(line_stream)
    if not (line) or not (line.startswith(initial_text)):
        raise Exception('Broken file')
    heads_split = [el for el in line.replace(initial_text, '').split('\t') if el != '']
    heads = []
    if len(heads_split) > 1: # exclude case with no heads
        for i in range(0, len(heads_split), 2):
            heads.append((heads_split[i], float(heads_split[i + 1])))
    return heads

def _next_no_cr(line_stream, default=None):
    """
    Adapter for next that removes the C.R.
    :param line_stream:
    :return:
    """
    res = next(line_stream, default)
    if res:
        res = res.replace('\n', '')
    return res

if __name__ == '__main__':
    evaluate_predictions(import_predictions_from_file)