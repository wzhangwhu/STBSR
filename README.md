# STBSR

STBSR identifies spatial domains from spatial transcriptomics counts and coordinates.
The paper model refines spatial and feature graphs, fuses three graph views and
optimizes one fixed objective throughout training. The objective combines scaled
cosine reconstruction error (SCE), relation binary cross-entropy (BCE) and
cross-view consistency (CCR). The loss plateau determines the training endpoint.

The primary readout applies row-wise L2 normalization to the decoded
mean-expression matrix before KMeans. The same fitted model also provides a
64-dimensional encoder readout for comparison. Each dataset uses its specified
graph construction rule and supplied number of domains.

## Repository structure

```text
STBSR/                          Model, input preparation, training and reproduction helpers
metric_utils/                   Clustering and spatial-boundary metrics
notebooks/
  run_stbsr_151672.ipynb
  run_stbsr_ARTISTA_Stage57_telencephalon.ipynb
benchmark/                      Archived paper scores and representative partitions
example_data/                   Example inputs already distributed with this repository
config.json                     The 20 paper datasets, model settings and recorded seeds
run_stbsr_20datasets.py          Benchmark runner
smoke_test.py                    Small end-to-end model check
requirements.txt
```

## Installation

The reference environment used Python 3.10 and PyTorch 2.7.0 with CUDA 12.8.

```bash
git clone https://github.com/wzhangwhu/STBSR.git
cd STBSR
python -m pip install -r requirements.txt
python smoke_test.py --device cpu
```

Install a PyTorch build compatible with the available CUDA driver for GPU runs.
To open the notebooks, install JupyterLab in the same environment.

```bash
python -m pip install jupyterlab
jupyter lab
```

## Reproduce the paper results

| Notebook | Default paper replay | Optional training |
|---|---|---|
| [DLPFC 151672](notebooks/run_stbsr_151672.ipynb) | Fig. 2a and 2c, 151672 ten-run results, markers and reconstructed representation | Joint-model run 5, recorded seed 11059586 |
| [ARTISTA Stage 57](notebooks/run_stbsr_ARTISTA_Stage57_telencephalon.ipynb) | Fig. 4a ARIs, recomputed Fig. 4c edge recovery, markers and ten-run reconstruction benchmark | Joint-model seed 46 with both readouts |

Both notebooks open in paper-replay mode. This mode recomputes ARIs from the
stored assignments and recreates the indicated panels and summaries. It does
not need a GPU or the raw training inputs. The Stage 57 map with ARI 0.6371 is
the historical encoder partition used in Fig. 4a. Current-model retraining
uses the fixed joint objective and produces a separate partition.

Set `STBSR_RUN_MODE=train` and `STBSR_DATA_ROOT` in the environment or the
notebook configuration cell to train a representative run. Set
`STBSR_DEVICE=cpu` for a CPU run. The saved assignments give exact display
replay across machines. Fresh training scores may vary across environments.

The [benchmark tables](benchmark/) include the ten final STBSR ARIs and
associated metrics for each of the 20 datasets, and the dataset-level mean
ARIs for all compared methods from Supplementary Data 1. Each dataset is given
equal weight in the overall mean. External-method training is outside this repository.

## Run the benchmark

Download the original datasets identified in the paper's Data Availability
section and place them under the relative paths in `config.json`.

- DLPFC and HBC use a Visium directory containing
  `filtered_feature_bc_matrix.h5` and `spatial/tissue_positions_list.csv`.
  Evaluation uses the accompanying `<section>_truth.txt` for DLPFC and
  the `ground_truth` column of `metadata.tsv` for HBC.
- Chicken heart and ARTISTA use raw-count h5ad files with `obsm["spatial"]`.
  The evaluation keys are `region` and `Annotation`, respectively. ARTISTA
  training reads the original `layers['counts']` matrices. The older bundled
  Stage 57 example contains transformed expression. Use the original
  ARTISTA h5ad for fresh training.

```bash
python run_stbsr_20datasets.py --data-root /path/to/data --dry-run
python run_stbsr_20datasets.py --data-root /path/to/data --output-root outputs/paper
```

The runner uses the recorded ten seeds per dataset, including the replacement
ten-run batch for 151672. Results are written to `per_run.csv` and
`by_dataset.csv`. The recorded endpoint replay devices are included in `config.json`. Use
`--datasets` and `--seeds` for a smaller run. Use a new or empty output directory.

## Train or replay a single case

```bash
python -m STBSR.cli train --dataset 151672 --data-root /path/to/data --device cuda --replay-device cpu --seed 11059586 --out outputs/151672
python -m STBSR.cli predict --dataset 151672 --data-root /path/to/data --device cpu --checkpoint outputs/151672/endpoint_model.pt --out outputs/151672_replay
python -m STBSR.cli score --predictions outputs/151672/predictions.npz --labels annotation.tsv
python -m STBSR.cli plot --predictions outputs/151672/predictions.npz --labels annotation.tsv --out outputs/151672_domains
```

`--input` accepts a single Visium directory or h5ad path instead of a
dataset root. The `score` and `plot` commands take a CSV or TSV with a header
and `spot_id,label` columns (configurable). Reference labels are read after
training and are aligned by observation ID before scoring.

A normal run saves the endpoint model, loss history, run metadata and both
KMeans assignments. `--save-reconstruction` also saves the decoded matrix.
`--compact` retains the two label arrays and omits the checkpoint and latent
matrix. Loss-stop status and the number of updates are recorded in `run.json`.
