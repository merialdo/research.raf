import collections

from adapter import adapter_factory
from model import datamodel, simulation
from model.bdsa_data import BdsaData
from pipeline.pipeline_abstract import AbstractPipeline
from pipeline.pipeline_common import ClusteringOutput
from utils import bdsa_utils, string_utils


class PipelineDataImport(AbstractPipeline):
    """
    Import external sources, generate some data that will be used for alignment/linkage
    """

    def __init__(self):
        spec_adapter = adapter_factory.spec_factory()
        self.sgen = spec_adapter.specifications_generator()
        self.sim = simulation.Simulation(spec_adapter)

    def run(self, data=None):
        """

        :param data:
        :return: (dict cat --> BdsaData, debug_stats)
        """
        output = collections.defaultdict(self.bdsa_creator)
        for source in self.sgen:
            if self.sim.do_add_source(source):
                output[source.category].add_source(source)
            del source
        del self.sgen
        res = {}
        for category, bdsa_data in output.items():
            bdsa_data.finalize_import()
            cl_output = ClusteringOutput()
            cl_output.bdsa_data = bdsa_data
            res[category] = cl_output
        # Currently this step has no debug_stats in output, so we return an empty dict
        return res, collections.defaultdict(dict)

    def name(self):
        return "ImportSources"

    def need_input(self):
        return False

    def need_output(self):
        return True

    def bdsa_creator(self):
        return BdsaData(self.sim, string_utils.folding_using_regex)