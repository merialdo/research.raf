# RAF Std approach

This is an implementation of the RaF-Std approach, for semantic type discovery using clustering.
RaF-StD takes as input a dataset of multiple sources providing entity descriptions in form of 
key-value pairs. It requires a partial (even noisy) linkage between sources.
## Installation

- Make sure Python 3.6+ is installed on your machine
- Clone the current repository in a chosen directory. We will call this directory now on --> $THIS_REPO
- From your project root, with your python installation (or within a venv or conda environment) launch 
`pip install -r requirements.txt`
- Copy file files_for_installation/local_config.ini to config/ directory, and adapt it as explained in file comments
  - we will refer from now with $INPUT_PATH as the directory you choose in this file to put your input data. Similary
  for all others parameters.
  - NB: remember to create the directory you provided in the config file, otherwise you will get unexpected errors.
- From the root project, launch the following functional tests to verify that your installation ended correctly: 
`python -m functional_tests.functional_test_launcher`. You should see in the output `Process finished with exit code 0`.

**NB**: if you encounter issues with python dependencies, you may miss the following packages:
 tkinter tk-devel, gcc, openssl-devel bzip2-devel. Try re-installing Python3.6 and follow those instructions 
 https://stackoverflow.com/questions/12344970/building-python-from-source-with-zlib-support and then install those packages.

## Retrieve data
- Extract all the datasets (ZIP files in $INPUT/dataset/*_dataset.zip) in a directory with the name
of the zip (e.g: $INPUT/dataset/camera_dataset.zip --> $INPUT/dataset/camera_dataset/ , ..)

## Launch

  - Open the root of the project
  - Launch:
  ```sh
$ python -m launcher.automatic_evaluation_launcher [mode] [exp_name]
```

* *[mode] is the algorithm you want to launch (RAF full, only source attribute matching, baselines...)*

| Algorithm to launch                                         | mode      | 
|-------------------------------------------------------------|-----------|
| Standard RaF-StD   (*)                                      | tag-p--7  | 
| Standard RaF-StD without name grouping  (*)                 | tag-n0--7 | 
| Vanilla Approach                                            | mixbl     |  
| Only Source Attribute Matching                              | classic   |     

**(\*)**: The number 7 means that after 7 iterations in which the stop condition is not met, 
the algorithm stops and returns an error.              

* *[exp_name] is the name of a set of experiments that you want to launch.
Each experiments defines the name of dataset, variations (limit % of linkage, nb of sources...) etc.*

A list of all predefined experiment is in the experiment_configuration directory in this repository.

| Experiments configuration                                   | exp_name              | 
|-------------------------------------------------------------|-----------------------|
| Whole DI2KG dataset (camera, monitor)                       | di2kg_full            | 
| Whole WDC dataset (automotive, jewelry, clothing)           | wdc_full              | 
| Whole DI2KG dataset, varying the match threshold            | di2kg_match_threshold |  
| Whole DI2KG dataset, varying the error rate                 | di2kg_error           |     
| Whole DI2KG dataset, varying the percentage of linkage  (*) | di2kg_linkage         |   
| DI2KG dataset, increasing number of sources             (*) | di2kg_scalability     |   
| Wdc dataset, varying the percentage of linkage          (*) | wdc_linkage           |  
| Wdc dataset, increasing number of sources               (*) | wdc_scalability       |  

(*): these experiments require choosing random sources or keep a random % of linkage.
To avoid biases, there are 5 copies for each experiment (random1, random2...)
It is suggested to compute an average of metrics (precision, recall, time) of all copies to get a good result.

You will find results of your experiments in evaluation_results.csv file in experiments directory.
  - *P*, *R*, *F1* provide precision, recall and F-masure, while *config* and *mode* the current configuration and mode.
  - Notice that the "time" column computes the whole experiments time but not the metrics (precision and recall)
  computation, which may be non negligible. 

Also, you will find a directory with the details of the output, among which:
- cluster_synthesis: list of clusters with the most frequent attribute names
- cluster_detail: list of source attributes associated to each cluster (this is before the last step that converts
  source attribute clusters to triple clusters).
- ikgpp: triple clusters, grouped by entity. The format is: entity_id --> triple_cluster_id --> triple
  - NB: notice that a cluster id appears under multiple entities. This is because each cluster refers to a semantic type
    and it is not specific to an entity. The cluster name is a code that contains the most frequent name found in its triples.
- a copy of the configuragtion used for the experiment
- log... file that contains time of each step.





