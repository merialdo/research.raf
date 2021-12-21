import collections

from tqdm import tqdm

from pipeline.pipeline_abstract import AbstractPipeline
from pipeline.pipeline_common import ClusteringOutput
from nltk.tokenize import word_tokenize
from nltk import ngrams

from utils.tagger_utils import Tagger

ATT_TAG = 'att_tags'

VAL_TAG = 'val_tags'

MAX_RATIO_OCCS = 0.05
MIN_NGRAM = 1
MAX_NGRAM = 4

class PipelineTagDebug(AbstractPipeline):
    """
    Pipeline step to tag top3 most frequent values of isolated attributes with atomic values. Used for debug
    """

    def __init__(self):
        self.aname2cname = None
        self.val2cname = None

    def run(self, data):
        result = {}
        self.debug_stats = data[1]
        for cat, bdsa_data in data[0].items():
            result[cat] = self.compute_ner(bdsa_data, cat)

        return result, self.debug_stats

    def name(self):
        return "PipelineNamedEntityRecognition"

    def need_input(self):
        return True

    def need_output(self):
        return True

    def compute_ner(self, output: ClusteringOutput, cat: str) -> ClusteringOutput:
        bdsa_data = output.bdsa_data
        tagger = self.tagger_factory(bdsa_data, output)

        # Now we replace the most frequent elements with a tag
        for sa in tqdm(output.sa_isolated.keys(), desc='Tagging isolated elements...'):
            sa_topvalues = bdsa_data.sa2topvalues[sa]
            for i, value in enumerate(sa_topvalues):
                sa_topvalues[i] = (tagger.tag(value[0]), value[1])
        return output

    def tagger_factory(self, bdsa_data, output):
        """
        We assign each value to a cluster, identified with the most frequent attribute name
        :param bdsa_data:
        :param output:
        :return:
        """
        val2cid = collections.defaultdict(set)
        aname2cid = collections.defaultdict(set)
        cid2most_frequent_name = {}
        for cid, source2sas in tqdm(output.sa_clusters.items(), desc='Build map with potential tags'):
            attribute_names = collections.Counter()
            for sas in source2sas.values():
                for sa in sas:
                    attribute_names[sa.name] += 1
                    aname2cid[tuple(word_tokenize(sa.name))].add(cid)
                    for value in bdsa_data.sa2value2occs[sa].keys():
                        # Keep only rare values and not too long
                        if bdsa_data.get_nb_attributes_value_occurs[value] / len(bdsa_data.sa2size) <= MAX_RATIO_OCCS \
                                and len(value) <= 25: #TODO 25
                            val2cid[value].add(cid)
            cid2most_frequent_name[cid] = attribute_names.most_common(1)[0][0]
        #True, false for distinguish between
        val2tag = {}
        for name in val2cid.keys() | aname2cid.keys():
            res = [(True, cid2most_frequent_name[cid]) for cid in val2cid[name]]
            res.extend((False, cid2most_frequent_name[cid]) for cid in aname2cid[name])
            val2tag[name] = res
        return Tagger(val2tag, _function_extractor, _function_joiner, False)


def _function_extractor(gram:list, elems:set):
    if not elems or len(elems) == 0:
        return ' '.join(gram)

    val_str = [name[1] for name in elems if name[0]]
    attname_str = [name[1] for name in elems if not name[0]]
    val_tag = "V[%s]" % ','.join(val_str) if len(val_str) > 0 else ''
    attname_tag = "A[%s]" % ','.join(attname_str) if len(attname_str) > 0 else ''
    return "##%s%s_%s##" % (val_tag, attname_tag, ' '.join(gram))


def _function_joiner(left: str, tag: str, right: str):
    return (left or '') + tag + (right or '')

        # >>> tg = Tagger({tuple(word_tokenize('bob carl it''s')): [1, 2]},\
        #     {tuple(word_tokenize('bob carl its')): [1], tuple(word_tokenize('bob carl')): [3],\
        #      tuple(word_tokenize('built-in')): [4]},\
        #      {1: 'EL1', 2: 'EL2', 3: 'EL3', 4: 'EL4'})
        # >>> tg.tag('alice bob carl it''s like built-in a23')
        # ('alice ##V[EL1,EL2]A[EL1]## like ##A[EL4]## a23', True)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
