
from external_tools.coma import coma
from model import dataset, datamodel
from model.bdsa_data import BdsaData
from pipeline.cluster_utils import WeightedEdge
from utils import io_utils
from config.bdsa_config import _config_

DOT_REPLACEMENT = '_dot_'


class ComaAdapter:

    def __init__(self, _bdsa_data: BdsaData):
        self.coma_adapter = coma.Coma(1, 'COMA_OPT_INST', '4096m')
        self.bdsa_data = _bdsa_data
        self.temporary_dir = _config_.get_cache_dir()

    def _generate_coma_format(self, sas, source_name):
        """
        Generate coma format for a specific SA
        """
        data = dataset.Dataset([sa.name for sa in sas] + ['url'])
        for page in set().union(*(self.bdsa_data.source2pages[sa.source] for sa in sas)):
            dict_values = {sa.name: self.bdsa_data.page2sa2value[page].get(sa, '') for sa in sas}
            dict_values['url'] = page.url
            data.add_row(dict_values)
        csv_filename = source_name.replace('.', DOT_REPLACEMENT)  # '%s_%s' % (sa.source.site, sa.name)
        data.export_to_csv(self.temporary_dir, csv_filename, False)
        return csv_filename

    def compare_sas_group(self, sas1, sas2, site1, site2, cat):
        file1 = self._generate_coma_format(sas1, site1)
        file2 = self._generate_coma_format(sas2, site2)
        match_name = 'match_%s_%s.txt' % (file1, file2)
        file1_with_ext = '%s.csv' % file1
        file2_with_ext = '%s.csv' % file2
        result_file = self.coma_adapter.run_coma_jar(file1_with_ext, file2_with_ext, match_name, self.temporary_dir)
        coma_res = io_utils.import_generic_file_per_line(result_file)
        # For some misterious reasons, sometimes coma puts in the output the full path of the directory (with '_'
        # instead of '/', and sometimes not. We have to take this into consideration.
        prefix_coma = self.temporary_dir[1:].replace('/', '_') + '_'

        # There are multiple rows
        results = []
        for row in coma_res:
            row_splitted = row.split(' <-> ')
            second_att_splitted = row_splitted[1].split(': ')
            first_att = row_splitted[0]
            second_att = second_att_splitted[0]
            score = float(second_att_splitted[1])
            if '.' in first_att:
                if prefix_coma in first_att:
                    first_att = first_att.replace(prefix_coma, '')
                if prefix_coma in second_att:
                    second_att = second_att.replace(prefix_coma, '')
                att1_split = first_att.split('.')
                att2_split = second_att.split('.')
                sa1_out = datamodel.source_attribute_factory(cat, att1_split[0].replace(DOT_REPLACEMENT, '.'),
                                                             att1_split[1])
                sa2_out = datamodel.source_attribute_factory(cat, att2_split[0].replace(DOT_REPLACEMENT, '.'),
                                                             att2_split[1])
                if sa1_out.name != 'url' and sa2_out.name != 'url':
                    results.append(WeightedEdge(sa1_out, sa2_out, score))

        # os.remove(result_file)
        return results

    def compare_sas(self, sa1, sa2, cat):
        res = self.compare_sas_group([sa1], [sa2], sa1.source.site, sa2.source.site, cat)
        score = 0
        for edge in res:
            if edge.node1 == sa1 and edge.node2 == sa2:
                score = edge.weight
        return score
