from enum import Enum

from pipeline import pipeline_compute_weights, pipeline_analyzer, pipeline_import_sources, baseline_name_rare_values, \
    pipeline_cluster_by_frequent_names, pipeline_tag, pipeline_clear_output, pipeline_cluster_by_name_full, \
    pipeline_significant_tokens, baseline_domains
from config.bdsa_config import _config_

class GroupByName(Enum):
    NONE = 1  # No group by name
    FULL = 2  # Group by name full (exclude isolated)
    SAME_PAGE = 3  # Group by name, but apply same-page rule (2 attributes in same page should never be together)
    EXCLUDE_SPLIT = 4 # Exclude from group by name original attributes split in multiple
    SAME_PAGE_EXCL_SPLIT = 5 # Apply SAME_PAGE and EXCLUDE_SPLIT
    BLACKLIST_SPLIT = 6 # Black list names from which someone extracted a virtual attribute

def build_tag_or_mi_iterations(tag, max_nb_steps, tag_tru_mi_false=True, group_by_name=GroupByName.FULL, comapp=False):
    """
    Generator for TAG or MI iterative pipeline
    Stop iterations of alignment - tagging when cluster do not change anymore. If we reach max_nb_steps without
    reaching the condition clause, then throw an error.
    :param tag:
    :param max_nb_steps:
    :param tag_tru_mi_false:
    :param group_by_name_internally:
    :param group_by_name_final_step:
    :return:
    """

    yield pipeline_import_sources.PipelineDataImport()
    # Singleton as it knows if we have to stop iterations
    tagging_step = pipeline_tag.PipelineTag() if tag_tru_mi_false else pipeline_significant_tokens.PipelineSignificantTokens()
    attribute_matching_step = pipeline_compute_weights.PipelineComputeWeights(_use_coma=comapp)
    stop_condition_reached = False
    for step in range(max_nb_steps):
        yield attribute_matching_step
        if not attribute_matching_step.continue_iterations:
            stop_condition_reached = True
            break
        yield tagging_step
    if not stop_condition_reached:
        raise Exception("Stop condition not met after %d iterations. Process failed." % max_nb_steps)

    if group_by_name != GroupByName.NONE:
        same_page = group_by_name in {GroupByName.SAME_PAGE, GroupByName.SAME_PAGE_EXCL_SPLIT}
        exclude_split = group_by_name in {GroupByName.SAME_PAGE_EXCL_SPLIT, GroupByName.EXCLUDE_SPLIT}
        blacklist_names = group_by_name == GroupByName.BLACKLIST_SPLIT
        yield pipeline_cluster_by_name_full.PipelineClusterByNameFull(same_page, exclude_split, blacklist_names)
    yield pipeline_clear_output.PipelineClearOutput()
    yield pipeline_analyzer.PipelineAnalyzer(tag, False)

def build_tag_updown_iterations(tag, nb_steps, final_group_by_name=True):
    yield pipeline_import_sources.PipelineDataImport()
    tagging_step = pipeline_tag.PipelineTag()
    compute_weights = pipeline_compute_weights.PipelineComputeWeights()
    yield from _external_iterations_tagging_alignment(compute_weights, nb_steps, tagging_step)
    _config_.switch_record_linkage_behavior()
    tagging_step.continue_iterations = True
    yield from _external_iterations_tagging_alignment(compute_weights, nb_steps, tagging_step)
    if final_group_by_name:
        yield pipeline_cluster_by_name_full.PipelineClusterByNameFull()
    yield pipeline_clear_output.PipelineClearOutput()
    yield pipeline_analyzer.PipelineAnalyzer(tag, False)

def _external_iterations_tagging_alignment(compute_weights, nb_steps, tagging_step):
    for step in range(nb_steps):
        yield compute_weights
        yield tagging_step
        if not tagging_step.continue_iterations:
            break
    if tagging_step.continue_iterations:
        yield pipeline_compute_weights.PipelineComputeWeights()


_pipelines = {
    'classic': lambda tag, nb: iter([
        pipeline_import_sources.PipelineDataImport(),
        pipeline_compute_weights.PipelineComputeWeights(),
        pipeline_analyzer.PipelineAnalyzer(tag, False)
    ]),
    'comapp': lambda tag, nb: iter([
        pipeline_import_sources.PipelineDataImport(),
        pipeline_compute_weights.PipelineComputeWeights(_use_coma=True, _debug_matching_score=True),
        pipeline_analyzer.PipelineAnalyzer(tag, False)
    ]),
    'classic-name': lambda tag, nb: iter([
        pipeline_import_sources.PipelineDataImport(),
        pipeline_compute_weights.PipelineComputeWeights(),
        pipeline_cluster_by_name_full.PipelineClusterByNameFull(),
        pipeline_analyzer.PipelineAnalyzer(tag, False)
    ]),
    'namebl': lambda tag, nb: iter([
        pipeline_import_sources.PipelineDataImport(),
        baseline_name_rare_values.PipelineComputeWeights(False),
        pipeline_analyzer.PipelineAnalyzer(tag, True)
    ]),
    'domainbl': lambda tag, nb: iter([
        pipeline_import_sources.PipelineDataImport(),
        baseline_domains.BaselineDomainAttributes(),
        pipeline_analyzer.PipelineAnalyzer(tag, True)
    ]),
    'mixbl': lambda tag, nb: iter([
        pipeline_import_sources.PipelineDataImport(),
        baseline_name_rare_values.PipelineComputeWeights(False),
        baseline_domains.BaselineDomainAttributes(True, _threshold_frequency=nb/100 if nb and nb != 0 else 0.2),
        pipeline_analyzer.PipelineAnalyzer(tag, True)
    ]),
    'namegroup': lambda tag, nb: iter([
        pipeline_import_sources.PipelineDataImport(),
        pipeline_compute_weights.PipelineComputeWeights(),
        pipeline_cluster_by_frequent_names.PipelineClusterByFrequentNames(),
        pipeline_analyzer.PipelineAnalyzer(tag, True)
    ]),
    'tag-n0': lambda tag, nb: build_tag_or_mi_iterations(tag, nb, True, GroupByName.NONE), # RaF-StD, excluding name grouping step
    'tag-p': lambda tag, nb: build_tag_or_mi_iterations(tag, nb, True, GroupByName.SAME_PAGE),  # RaF-StD standard
    'tag-c': lambda tag, nb: build_tag_or_mi_iterations(tag, nb, True, GroupByName.SAME_PAGE, True),  # RaF-StD replacing SAM by COMA
    'mi-n0': lambda tag, nb: build_tag_or_mi_iterations(tag, nb, False, GroupByName.NONE),  # Mutual information without name grouping
    'mi': lambda tag, nb: build_tag_or_mi_iterations(tag, nb, False, GroupByName.FULL),  # Mutual information

    ## OLD and tests
    'tag': lambda tag, nb: build_tag_or_mi_iterations(tag, nb, True, GroupByName.FULL),  # Tagging, group attributes by name at final step
    'tag-x': lambda tag, nb: build_tag_or_mi_iterations(tag, nb, True, GroupByName.EXCLUDE_SPLIT),  # Tagging, group attributes by name at final step
    'tag-b': lambda tag, nb: build_tag_or_mi_iterations(tag, nb, True, GroupByName.BLACKLIST_SPLIT),  # Tagging, group attributes by name at final step
    'tag-xp': lambda tag, nb: build_tag_or_mi_iterations(tag, nb, True, GroupByName.SAME_PAGE_EXCL_SPLIT),  # Tagging, group attributes by name at final step
    'tag-ud': lambda tag, nb: build_tag_updown_iterations(tag, nb, True), # Tagging, when iteration stop then go upside down
    'tag-ud0': lambda tag, nb: build_tag_updown_iterations(tag, nb, False)
}

