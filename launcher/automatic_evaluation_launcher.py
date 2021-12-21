import collections
import getopt
import os, sys, glob
import ntpath
import re
import traceback
from configparser import ConfigParser, ExtendedInterpolation

from config.bdsa_config import _config_
import shutil

import project_constants
from launcher import bdsa_launcher
from scripts import results_evaluation
from scripts.results_evaluation import SchemaLevelEvaluation
from utils import string_utils, io_utils
from utils.bdsa_utils import get_git_commit, get_git_commit_message

PARTIAL_EVALUATION_CSV = 'partitioned_evaluation.csv'

ERROR_MESSAGE = 'Syntax: python -m launcher.automatic_evaluation_launcher [--debug] mode1,mode2,mode3'

OLD = 'OLD'

LAUNCH_FIELDS = ['config', 'mode', 'git_commit_message',
                 'P', 'R', 'F1',
                 'IL_TP', 'timestamp', 'launch_label', 'commit_id', 'dataset',
                 'ground_truth', 'total_time']

PARTITIONED_EVALUATION_FIELDS = ['launch_label', 'partition_tag', 'Pw', 'Rw', 'git_commit_message', 'timestamp']

OUTPUT_CSV_FILE = 'evaluation_results.csv'

"""
Launches algorithm with different configurations, evaluate them against a ground truth, and put the result in an output file
"""

OUTPUT_DIRECTORY = _config_.get_experiments_output_dir()
PARAMETERS_REPO_PATH = _config_.get_algo_parameters_repository_path()

AlgorithmLaunch = collections.namedtuple('AlgorithmLaunch', LAUNCH_FIELDS)
PartitionedEvaluation = collections.namedtuple('PartitionedEvaluation', PARTITIONED_EVALUATION_FIELDS)

CALCULATE_NON_WEIGHTED_PR = False


class AlgoMode:
    def __init__(self, name, nb=None):
        self.name = name
        self.nb = int(nb) if nb else None

    def __str__(self):
        return self.name if not self.nb else '%s--%d' % (self.name, self.nb)


def re_evaluate(parameters_repo_subdir: str = None, ground_truth_file=None):
    """
    Re-evaluate metrics for all experiments OR only experiments of provided config dir
    :param parameters_repo_subdir:
    :param ground_truth_file:
    :return:
    """

    ground_truth_label = ntpath.basename(ground_truth_file) if ground_truth_file else 'default'
    config_reader = ConfigParser(interpolation=ExtendedInterpolation())

    directories_path = _get_all_dirs_to_reanalyze(parameters_repo_subdir)
    for dir_path in directories_path:
        config_files = glob.glob(os.path.join(dir_path, '*.ini'))
        if len(config_files) == 0:
            continue
        config_file = config_files[0]
        config_reader.read([config_file])
        config_name = ntpath.basename(config_file).replace('.ini', '')
        dataset = config_reader.get('inputs', 'specifications')
        dir = ntpath.basename(dir_path)
        template = re.search('^([^_]+)_.*_git-([a-z0-9]+)', dir, re.IGNORECASE)
        if template:
            git_commit_id = template.group(2)
            git_message = get_git_commit_message(git_commit_id)
            _evaluate_launch(config_name, dataset, git_commit_id, git_message, ground_truth_file, ground_truth_label,
                             dir, mode=template.group(1), output_dir=os.path.join(OUTPUT_DIRECTORY, dir_path))


def _get_all_dirs_to_reanalyze(parameters_repo_subdir):
    # Define which launches (ie which directories) we take into account
    directories = set()
    if parameters_repo_subdir:
        parameter_directory = os.path.join(PARAMETERS_REPO_PATH, parameters_repo_subdir)
        for aconfig_filename in os.listdir(parameter_directory):
            aconfig = aconfig_filename.replace('.ini', '')
            launch_dir_pattern = '*_%s-*_git-*' % aconfig
            directories.update(glob.glob(os.path.join(OUTPUT_DIRECTORY, launch_dir_pattern)))
    else:
        directories.update(glob.glob(os.path.join(OUTPUT_DIRECTORY, '*_git-*')))
    return directories


def launch(parameters_repo_subdir: str, modes: list, ground_truth_file=None, relaunch=False):
    """

    :param parameters_repo_subdir: subdirectory of PARAMETERS_REPO_PATH in which ini parameters files can be found. If there
    are more than one ini file, all of them will be used
    :param modes: list of pair tuples (mode, param (if any))
    :param relaunch: if true, launch algorithm in any case. If false, and a launch with same id, mode and conf already exists, then just re-evaluate it.
    :return:
    """
    # TODO redirect out vs file? cf. https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
    git_commit_id, git_commit_message = get_git_commit()
    ground_truth_label = ntpath.basename(ground_truth_file) if ground_truth_file else 'default'

    # For each configuration file
    parameter_directory = os.path.join(PARAMETERS_REPO_PATH, parameters_repo_subdir)
    for aconfig_filename in os.listdir(parameter_directory):
        config_path = os.path.join(parameter_directory, aconfig_filename)
        config_hash = io_utils.compute_file_hash(config_path)[:6]
        shutil.copyfile(config_path, os.path.join(project_constants.ROOT_DIR, 'config', 'algo_parameters.ini'))
        _config_.reset_config()
        dataset_name = ntpath.basename(_config_.get_specifications())
        for mode in modes:
            warnings = []
            aconfig = aconfig_filename.replace('.ini', '')

            # A label given to this particular launch, that takes into account the commit, config and launch mode
            launch_label = '%s_%s-%s_git-%s' % (str(mode), aconfig, config_hash, git_commit_id)
            try:
                output_dir, output_dir_existed_and_was_nonempty = _build_output_dir(launch_label, relaunch)
                if not output_dir_existed_and_was_nonempty or relaunch:
                    total_time = bdsa_launcher.launch_bdsa(mode.name, 0, 0, launch_label, mode.nb)
                else:  # If it it a relaunch, then we get total time from output
                    print("chow")
                    try:
                        with open(glob.glob(os.path.join(output_dir, "log_*"))[0], 'r') as file:
                            lines = file.readlines()
                            last_line = lines[-1].split(" ")
                            total_time = float(last_line[-1])
                            print(last_line)
                    except:
                        total_time = -1
                output_dir = _move_results_to_output(output_dir, config_path, launch_label)

                warnings.extend(
                    _evaluate_launch(aconfig, dataset_name, git_commit_id, git_commit_message, ground_truth_file,
                                     ground_truth_label, launch_label, mode, output_dir, total_time))
                if len(warnings) > 0:
                    io_utils.output_txt_file(warnings, 'warnings.txt', output_dir)
            except Exception as err:
                launch_err = AlgorithmLaunch(launch_label=launch_label, config='ERR ' + aconfig, mode=str(mode),
                                             commit_id=git_commit_id,
                                             git_commit_message='ERR: %s -- %s' % (str(err), git_commit_message),
                                             timestamp=string_utils.timestamp_string_format(),
                                             dataset=dataset_name, ground_truth=ground_truth_label, P=-1, R=-1, F1=-1,
                                             IL_TP=-1, total_time=0)
                io_utils.append_csv_file(LAUNCH_FIELDS, [launch_err._asdict()], OUTPUT_CSV_FILE, OUTPUT_DIRECTORY)
                print(str(err))
                traceback.print_tb(err.__traceback__)


def _evaluate_launch(aconfig, dataset_name, git_commit_id, git_commit_message, ground_truth_file, ground_truth_label,
                     launch_label, mode, output_dir, total_time):
    """
    Evaluate a particular launch, put in CSV file
    :param aconfig:
    :param dataset_name:
    :param git_commit_id:
    :param git_commit_message:
    :param ground_truth_file:
    :param ground_truth_label:
    :param launch_label:
    :param mode:
    :param output_dir:
    :return:
    """
    warnings = []
    launch, partitioned_data = _create_launch_stats_with_evaluation(aconfig, dataset_name, git_commit_id,
                                                                    git_commit_message,
                                                                    ground_truth_file, ground_truth_label, launch_label,
                                                                    mode,
                                                                    output_dir, total_time)
    # Put ground truth comparison file in the output directory
    current_gtcomparison_file = get_latest_file(_config_.get_output_dir(),
                                                '%s_*' % results_evaluation.GROUND_TRUTH_COMPARISON_FILE_PREFIX)
    if current_gtcomparison_file:
        last_gtcomparison_file = get_latest_file(output_dir,
                                                 '%s_*' % results_evaluation.GROUND_TRUTH_COMPARISON_FILE_PREFIX)
        if last_gtcomparison_file:
            os.remove(last_gtcomparison_file)
        shutil.move(current_gtcomparison_file, output_dir)
    else:
        print("WARNING: no gt comparison build")
        warnings.append('NO GT comparison found')
    # Add data
    io_utils.append_csv_file(LAUNCH_FIELDS, [launch._asdict()], OUTPUT_CSV_FILE, OUTPUT_DIRECTORY)
    if len(partitioned_data) > 0:
        io_utils.append_csv_file(PARTITIONED_EVALUATION_FIELDS, partitioned_data, PARTIAL_EVALUATION_CSV,
                                 OUTPUT_DIRECTORY)
    return warnings


def _build_output_dir(launch_label, relaunch):
    """
    Build output dir if not exists. If exists and relaunch is true, move old files to OLD
    :param launch_label:
    :param relaunch:
    :return: true if already exists and is nonempty.
    """
    output_dir = os.path.join(OUTPUT_DIRECTORY, launch_label)
    output_dir_existed = os.path.exists(output_dir)
    output_dir_existed_and_was_nonempty = output_dir_existed and len(os.listdir(output_dir)) > 0
    if not output_dir_existed:
        os.mkdir(output_dir)
    # If dir already exists (ie a launch with exactly same config and code was launched)
    # and relaunch is activated, then move old results to OLD
    else:
        if relaunch:
            old_dir = os.path.join(OUTPUT_DIRECTORY, OLD)
            for file in os.listdir(output_dir):
                try:
                    shutil.move(os.path.join(output_dir, file), old_dir)
                except Exception as e:
                    print(e)
    return output_dir, output_dir_existed_and_was_nonempty


def _create_launch_stats_with_evaluation(aconfig, dataset_name, git_commit_id, git_commit_message, ground_truth_file,
                                         ground_truth_label,
                                         launch_label, mode, output_dir, total_time):
    """
    Evaluate results and create objects with all data about the launch
    :param aconfig: the config name
    :param dataset_name: the dataset name
    :param git_commit_id:
    :param git_commit_message:
    :param ground_truth_file:
    :param ground_truth_label:
    :param launch_label: full launch label name
    :param mode: with/without tag, baseline...
    :param output_dir: the current output dir
    :return:
    """
    # Analyze P-R normal (if asked) and weighted
    detail_file = get_latest_file(output_dir, 'cluster_detail_%s*' % launch_label)
    ikgpp_file = get_latest_file(output_dir, 'ikgpp_%s*' % launch_label)
    evaluator = results_evaluation.ResultsEvaluator()
    results = evaluator.launch_evaluation_files(detail_file, ikgpp_file, do_partitioned_evaluation=True,
                                                output_gt_comparison=True,
                                                schema_level_evaluation=SchemaLevelEvaluation.NONE)

    # if CALCULATE_NON_WEIGHTED_PR:
    #     results_non_weighted = evaluator. \
    #         launch_evaluation_files(detail_file, None, do_partitioned_evaluation=False, output_gt_comparison=False,
    #                                 schema_level_evaluation=SchemaLevelEvaluation)
    #     precision_non_weighted = results_non_weighted.schema.precision
    #     recall_non_weighted = results_non_weighted.schema.recall
    # else:
    #     precision_non_weighted = -1
    #     recall_non_weighted = -1
    # Create object with eval
    launch = AlgorithmLaunch(launch_label=launch_label, config=aconfig, mode=str(mode),
                             commit_id=git_commit_id, git_commit_message=git_commit_message,
                             timestamp=string_utils.timestamp_string_format(),
                             P=results.instance_lib.precision, R=results.instance_lib.recall,
                             F1=2 * (results.instance_lib.precision * results.instance_lib.recall) /
                                (results.instance_lib.precision + results.instance_lib.recall),
                             IL_TP=results.instance_lib.true_positives,
                             dataset=dataset_name, ground_truth=ground_truth_label, total_time=total_time)
    # Compute partitioned
    partitioned_data = [PartitionedEvaluation(launch.launch_label, partition_name, partition_eval.precision,
                                              partition_eval.recall, launch.git_commit_message,
                                              launch.timestamp)._asdict()
                        for partition_name, partition_eval in results.instance_lib.subsets.items()]
    return launch, partitioned_data


def _move_results_to_output(output_dir, config_path, launch_label):
    """
    Move results to specific directory.
    :param config_path: path of config file
    :param launch_label:
    :return: the output dir
    """

    # Move config and all output data to the output directory
    # https://stackoverflow.com/questions/11835833/why-would-shutil-copy-raise-a-permission-exception-when-cp-doesnt
    shutil.copyfile(config_path, os.path.join(output_dir, os.path.basename(config_path)))
    for file in glob.glob(os.path.join(_config_.get_output_dir(), '*%s*' % launch_label)):
        shutil.move(file, output_dir)
    return output_dir


def get_latest_file(directory, pattern):
    """
    Return latest file with provided pattern, or null if none were found
    :param directory:
    :param pattern:
    :return:
    """
    all_files = glob.glob(os.path.join(directory, pattern))
    return max(all_files, key=os.path.getctime, default=None)


if __name__ == '__main__':
    # https://www.tutorialspoint.com/python/python_command_line_arguments.htm
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ['debug', 'relaunch', 'eval'])
    except getopt.GetoptError:
        print(ERROR_MESSAGE)
        sys.exit(2)
    relaunch = False
    reevaluate = False
    for opt, arg in opts:
        if opt == '--debug':
            _config_.activate_debug()
        if opt == '--relaunch':
            relaunch = True
        if opt == '--eval':
            reevaluate = True
        else:
            print(ERROR_MESSAGE)
            sys.exit()
    if reevaluate:
        re_evaluate(args[1] if len(args) > 0 else None, None)
    else:
        modes = []
        for mode in args[0].split(','):
            mode_arr = mode.split('--')
            modes.append(AlgoMode(mode_arr[0], mode_arr[1] if len(mode_arr) > 1 else None))
        launch(args[1], modes, None, relaunch)
