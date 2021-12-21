import os
import shutil

from appJar import gui

import project_constants
from config.bdsa_config import _config_
from gui.operation_launcher_form import OperationLauncherForm
from launcher import bdsa_launcher
from pipeline import pipeline_factory
from scripts import debug_alignment, results_evaluation
from scripts.results_evaluation import SchemaLevelEvaluation


class MainAlgorithmLauncher(OperationLauncherForm):
    """
    Main algorithm
    """

    def __init__(self):
        super().__init__('Main algorithm')
        self.launch_tag = 'Launch tag'
        self.config_file = 'External config file'
        self.external_config = 'Load external config'
        self.launch_mode = 'Launch mode'
        self.nb = 'Number of iterations (if applies)'


    def build_form(self, app: gui):
        app.addCheckBox(self.external_config)
        app.addLabelFileEntry(self.config_file)
        app.setEntry(self.config_file, _config_.get_algo_parameters_repository_path())
        app.addLabelEntry(self.launch_tag)
        app.addLabelOptionBox(self.launch_mode, pipeline_factory._pipelines.keys())
        app.addSpinBoxRange(self.nb, 0, 12)

    def extract_parameters_and_launch(self, app:gui):
        if app.getCheckBox(self.external_config):
            config_path = app.getEntry(self.config_file)
            shutil.copyfile(config_path, os.path.join(project_constants.ROOT_DIR, 'config', 'algo_parameters.ini'))
            _config_.reset_config()
        tag = app.getEntry(self.launch_tag)
        mode = app.getOptionBox(self.launch_mode)
        nb = int(app.getSpinBox(self.nb))
        bdsa_launcher.launch_bdsa(mode, 0,0, tag=tag, nb=nb)

class EvaluationLauncher(OperationLauncherForm):
    """
    Compute precision recall on the output of a file, provided a ground truth.
    See debug_alignment.evaluate_alignment_against_ground_truth
    """

    def __init__(self):
        super(EvaluationLauncher, self).__init__('Evaluation')
        self.ground_truth = 'Ground truth'
        self.output = 'Output'
        self.ikgpp = 'IKGPP'
        self.options = 'Options'
        self.slw = 'Schema-level weighted'
        self.sl = 'Schema-level non-weighted'
        self.il = 'Instance-level'
        self.ilp = 'Instance-level partitioned'
        self.export_results = 'Export results'
        self.results = 'Results'
        self.detailed_analysis_results = 'Results detailed'
        self.evaluator = results_evaluation.ResultsEvaluator()

    def build_form(self, app:gui):
        app.addLabelFileEntry(self.ground_truth)
        app.setEntryDefault(self.ground_truth, 'Defaulted to GT of dataset specified in parameters')
        app.setEntry(self.ground_truth, _config_.get_ground_truth_path())
        #app.addLabel("If not provided, default ground truth is the one for dataset provided in specifications parameter")
        app.addLabelFileEntry(self.output)
        app.addLabelFileEntry(self.ikgpp)
        app.addRadioButton(self.options, self.slw)
        app.addRadioButton(self.options, self.sl)
        app.addRadioButton(self.options, self.il)
        app.addRadioButton(self.options, self.ilp)
        app.addCheckBox(self.export_results)
        app.addLabel(self.results)
        app.addEmptyMessage(self.detailed_analysis_results)


    def extract_parameters_and_launch(self, app:gui):
        gt = app.getEntry(self.ground_truth)
        output = app.getEntry(self.output)
        ikgpp = app.getEntry(self.ikgpp)
        options = app.getRadioButton(self.options)
        do_comparison = app.getCheckBox(self.export_results)
        if options == self.il:
            do_partitioned = False
            sl = SchemaLevelEvaluation.NONE
        elif options == self.ilp:
            do_partitioned = True
            sl = SchemaLevelEvaluation.NONE
        elif options == self.sl:
            do_partitioned = False
            sl = SchemaLevelEvaluation.NON_WEIGHTED
        elif options == self.slw:
            do_partitioned = False
            sl = SchemaLevelEvaluation.WEIGHTED

        evaluation_results = self.evaluator.\
            launch_evaluation_files(output, ikgpp, sl, do_partitioned,
                                    output_gt_comparison=do_comparison)

        app.setLabel(self.results, 'Schema: %s\n Instance: %s' %
                     (str(evaluation_results.schema), str(evaluation_results.instance_lib)))
        if do_partitioned:
            print('\n'.join('%s --> %s' % (key, str(metrics)) for key, metrics in evaluation_results.instance.subsets.items()))
        #print (str(evaluation_results))



class ClusterDetailSimplifier(OperationLauncherForm):
    """
    Compute precision recall on the output of a file, provided a ground truth.
    See debug_alignment.evaluate_alignment_against_ground_truth
    """

    def __init__(self):
        super(ClusterDetailSimplifier, self).__init__('ClusterDetailSimplifier')
        self.output = 'OutputCDS'
        self.results = 'ResultsCDS'

    def build_form(self, app:gui):
        app.addLabelFileEntry(self.output)
        app.addLabel(self.results)

    def extract_parameters_and_launch(self, app:gui):
        app.setLabel(self.results, 'Pending...')
        output = app.getEntry(self.output)
        debug_alignment.build_simplified_cluster_detail_output(output)
        app.setLabel(self.results, 'OK')

class LoadConfig(OperationLauncherForm):
    """
    Load external config
    """

    def __init__(self):
        super().__init__('Load config')
        self.config_file = 'Config file to import'


    def build_form(self, app: gui):
        app.addLabelFileEntry(self.config_file)
        app.setEntry(self.config_file, _config_.get_algo_parameters_repository_path())

    def extract_parameters_and_launch(self, app:gui):
        config_path = app.getEntry(self.config_file)
        shutil.copyfile(config_path, os.path.join(project_constants.ROOT_DIR, 'config', 'algo_parameters.ini'))
        _config_.reset_config()


class MultiEvaluationLauncher(OperationLauncherForm):
    """
    Compute precision recall on the output of 2 files, provided a ground truth.
    See debug_alignment.evaluate_alignment_against_ground_truth
    """

    def __init__(self):
        super(MultiEvaluationLauncher, self).__init__('MultiEvaluationLauncher')
        self.ground_truth = 'M/Ground truth'
        self.solution1 = 'Solution1'
        self.solution2 = 'Solution2'
        self.weighted = 'M/Weighted'
        self.export_results = 'M/Export results'
        self.results = 'M/Results'

    def build_form(self, app:gui):
        app.addLabelFileEntry(self.ground_truth)
        app.setEntry(self.ground_truth, _config_.get_ground_truth_path())
        #app.addLabel("If not provided, default ground truth is the one for dataset provided in specifications parameter")
        app.addLabelFileEntry(self.solution1)
        app.addLabelFileEntry(self.solution2)
        app.addCheckBox(self.weighted)
        app.addCheckBox(self.export_results)
        app.addLabel(self.results)


    def extract_parameters_and_launch(self, app:gui):
        app.setLabel(self.results, 'Computing...')
        gt = app.getEntry(self.ground_truth)
        sol1 = app.getEntry(self.solution1)
        sol2 = app.getEntry(self.solution2)
        print('scripts.results_evaluation.compare_different_alignment_solutions(\
            %s, %s, %s, do_output_comparison=%s' % (sol1, sol2, gt, app.getCheckBox(self.export_results)))
        # scripts.results_evaluation.compare_different_alignment_solutions(
        #     sol1, sol2, gt, do_output_comparison=app.getCheckBox(self.export_results))
        app.setLabel(self.results, 'Exported.')
