# Identifying homogeneous subgroups of patients and important features: a topological machine learning approach

Ewan Carr<sup>1</sup>  
Mathieu Carrière<sup>2</sup>  
Bertrand Michel<sup>3</sup>  
Fred Chazal<sup>4</sup>  
Raquel Iniesta<sup>1</sup>  

<sup>1</sup> Department of Biostatistics and Health Informatics,
Institute of Psychiatry, Psychology \& Neuroscience, King's College London, United Kingdom.  
<sup>2</sup>Inria Sophia-Antipolis, DataShape team, Biot, France.  
<sup>3</sup>Ecole Centrale de Nantes, LMJL -- UMR CNRS 6629, Nantes, France.  
<sup>4</sup>Inria Saclay, Ile-de-France, France.  

For more information, please see the 🔓 paper in *BMC Bioinformatics*:

> Carr, E., Carrière, M., Michel, B. et al. Identifying homogeneous subgroups
> of patients and important features: a topological machine learning approach.
> *BMC Bioinformatics* **22**, 449 (2021). <https://doi.org/10.1186/s12859-021-04360-9>

Please contact <mailto:raquel.iniesta@kcl.ac.uk> for queries.

## About

This repository provides a pipeline for clustering based on topological data
analysis:

> **Background**
> This paper exploits recent developments in topological data analysis to present
> a pipeline for clustering based on Mapper, an algorithm that reduces complex
> data into a one-dimensional graph.
> 
> **Results**
> We present a pipeline to identify and summarise clusters based on statistically
> significant topological features from a point cloud using Mapper.
> 
> **Conclusions**
> Key strengths of this pipeline include the integration of prior knowledge to
> inform the clustering process and the selection of optimal clusters; the use of
> the bootstrap to restrict the search to robust topological features; the use of
> machine learning to inspect clusters; and the ability to incorporate mixed data
> types. Our pipeline can be downloaded under the GNU GPLv3 license at
> <https://github.com/kcl-bhi/mapper-pipeline>.


## Software

Our pipeline is written in Python 3 and builds on several open source packages,
including [`sklearn-tda`](https://github.com/MathieuCarriere/sklearn-tda),
[`GUDHI`](https://gudhi.inria.fr/python/latest/),
[`xgboost`](https://xgboost.readthedocs.io/en/latest/), and
[`pygraphviz`](https://pygraphviz.github.io/).

To get started, clone this repository and create a new virtual environment:

```
git clone https://github.com/kcl-bhi/mapper-pipeline.git
cd mapper-pipeline
python3 -m venv env
source env/bin/activate
pip install -r requirments.txt
```

## Using the pipeline

The pipeline expects an input dataset in CSV format. In our application, the
dataset contained information on ≈140 variables for ≈430 participants. If
`input.csv` is not found in the working directory, sample data will be
simulated.

The scripts should be used as follows:

1. **Generate input files** 

   ```{bash}
   python3 prepare_inputs.py
   ```
    1. Load the input dataset.
    2. Construct the Gower distance matrix. Note that this requires 
       categorical variables to be specified using `categorical_items.csv`.
    3. Define sets of parameters to explore via grid search.
    4. Create a dictionary containing all combinations of input parameters.

    ```python
    params = [{'fil': f,
               'res': r,
               'gain': gain}
              for f in fil.items()
              for r in resolutions
              for gain in [0.1, 0.2, 0.3, 0.4]]
     ```

    5. Store each set of inputs, and other required data, in the `inputs`
       directory.

2. **Run Mapper for each set of input parameters**

   The script `test_single_graph.py` runs Mapper for a single set of
   parameters. It requires three arguments:

   ```{bash}
   python3 test_single_graph.py '0333' 'inputs' 'outputs'
   ```

   `0333` refers to the set of parameters to test; `inputs` and `outputs`
   specify the folders to load inputs and save outputs. This script:

    1. Runs Mapper for the specified parameters (using `MapperComplex`).
    2. Identifies statistically significant, representative, topological features.
    3. Extracts required summaries and stores in the `outputs` subfolder.

3. **Process all outputs and produce summaries**

   ```{bash}
   python3 process_outputs.py
   ```

   This file:

   1. Loads all outputs (from `outputs`)
   2. Excludes graphs with no significant features or duplicate graphs.
   3. Splits each graph into separate topological features and removes features 
      with <5% or >95% of the sample.
   4. Derives required summaries for each feature. This includes homogeneity
      among feature members with respect to pre-specified outcome.
   5. Ranks all features by homogeneity and select the top N features.
   6. Visualise each top-ranked feature and output summaries to spreadsheet.

### Parallel computing

The grid search can be time-consuming, especially as the number of parameters
settings increase. Fortunately, this process can be [straightforwardly
parallelised](https://en.wikipedia.org/wiki/Embarrassingly_parallel) either
using multiple cores on a local machine or using cluster computing.

On a single machine, using `parallel` :  

```{bash}
python3 prepare_inputs.py
parallel --progress -j4 < jobs
```

On a cluster:

```{bash}
#!/bin/bash
#SBATCH --tasks=8
#SBATCH --mem=4000
#SBATCH --job-name=array
#SBATCH --array=1-2000
#SBATCH --output=logs/%a.out
#SBATCH --time=0-72:00
count=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
python3 test_single_graph.py $count "inputs" "outputs"
```
