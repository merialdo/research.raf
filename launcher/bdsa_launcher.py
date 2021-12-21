import time

from sys import argv

from pipeline import pipeline_factory
from utils import io_utils
from config.bdsa_config import _config_

"""
Main class for launching the schema alignment pipeline.
Example input: 1 3 test ==> Launch pipeline steps from 1 to 3 and tag results with code 'test'.

Cf. pipeline_factory.py for details on the pipeline

"""

__author__ = 'fpiai'

def launch_bdsa(pipeline:str, start, end, tag, nb):
    """
    Launch the pipeline from start to end
    :param start: initial step index
    :param end: final step index (included), or zero if go until the last possible step
    :param tag: 
    :return: 
    """
    if end != 0 and end < start:
        raise Exception("Start step should be before end step")

    time_file = []

    now = time.time()
    steps_iterator = pipeline_factory._pipelines[pipeline](tag, nb)
    current_step_number = 0

    # Skip steps until reach the start step
    for i in range(start):
        next(steps_iterator)
        current_step_number += 1

    print("Start pipeline")
    pipeline_current = next(steps_iterator)
    first_input = _import_first_step_if_necessary(pipeline_current, start, tag)

    step_import = time.time()
    padd('Data imported, took %f seconds'%(step_import - now), time_file)

    intermediate_data = pipeline_current.run(first_input)
    step_time = time.time()
    padd('First step (%s) run, took %f seconds' % (pipeline_current.name(), step_time - step_import), time_file)

    for pipeline_current in steps_iterator:
        current_step_number += 1
        if end != 0 and current_step_number > end:
            break
        intermediate_data = pipeline_current.run(intermediate_data)
        step_time_new = time.time()
        padd('End of step %s, took %f seconds, total %f seconds' % (pipeline_current.name(), step_time_new - step_time, step_time_new - now), time_file)
        step_time = step_time_new

    last_step = pipeline_current
    _output_last_step_if_necessary(end, last_step, tag, intermediate_data)
    step_new = time.time()
    total_time = step_new - now
    padd('End, last step %f, total time %f' % (step_new - step_time, total_time), time_file)
    io_utils.output_file_generic(time_file, lambda x: str(x), "%s_%s" % ('log', tag))
    return total_time

def padd(text, datalist:list):
    """
    Append time text to output and list that will be put in file
    :param text:
    :param datalist:
    :return:
    """
    print(text)
    datalist.append(text)

def _output_last_step_if_necessary(end, pipe_last, tag, last_output):
    """
    Output the results of the last step until now as cached results. Could be used later to relaunch step N+1
    :param end: 
    :param pipe_last: 
    :param tag: 
    :return: 
    """
    if pipe_last.need_output():
        for cat, output in last_output.items():
            fname = io_utils._build_filename("Step%s_%s_%s__" % (end, tag, cat), 'ser', _config_.get_cache_dir())
            io_utils.output_ser_file(output, fname)


def _import_first_step_if_necessary(pipeline_current, start, tag):
    """
    For a pipeline step N, imports the cached results of an old launch of the pipeline, until the step N-1.
    Only useful if we are not starting at step 1
        
    :param pipeline_current: 
    :param start: 
    :param tag: 
    :return: 
    """
    if pipeline_current.need_input():
        cat2filename = io_utils.find_files_pattern(_config_.get_cache_dir(), "Step%s_%s_" % (start-1, tag), '__', 'ser')
        cat2graphs = {}
        for cat, filename in cat2filename.items():
            cat2graphs[cat] = io_utils.import_ser_file(filename)
        return cat2graphs
    else:
        return None


if __name__ == "__main__":
    pipeline_nb = argv[1] if len(argv) >= 2 else 'classic'
    pipeline_nb_arr = pipeline_nb.split('--')
    pipeline = pipeline_nb_arr[0]
    nb = int(pipeline_nb_arr[1]) if len(pipeline_nb_arr) > 1 else 1
    tag = argv[2] if len(argv) >= 3 else 'default_name'
    start = int(argv[3]) if len(argv) >= 4 else 0
    end = int(argv[4]) if len(argv) >= 5 else 0
    print('Running pipeline %s with tag %s' % (pipeline_nb, tag))
    if start != 0 or end != 0:
        print('Only steps from %d to %d'%(start, end))
    launch_bdsa(pipeline, start, end, tag, nb)

