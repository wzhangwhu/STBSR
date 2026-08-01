# STBSR

STBSR is a spatial transcriptomics method for spatial domain identification.
This repository contains the method implementation and a reproducible benchmark
runner for 31 datasets.

## Method

STBSR combines spatial, feature, and fusion graphs with adaptive
representation fusion. Boundary-aware graph construction, cross-view
consistency, embedding regularization, and ZINB reconstruction are optimized
jointly to recover spatial domains while preserving tissue boundaries.

## Repository structure

```text
stbsr_code/
|-- STBSR/
|   |-- layers.py
|   |-- models.py
|   |-- pipeline.py
|   `-- utils.py
|-- metric_utils/
|   `-- benchmark_metric_utils.py
|-- example_data/
|   |-- ARTISTA_Stage57_telencephalon/
|   `-- 151671/
|-- notebooks/
|   |-- run_stbsr_ARTISTA_Stage57_telencephalon.ipynb
|   `-- run_stbsr_151671.ipynb
|-- datasets_31.csv
|-- requirements.txt
`-- run_stbsr_31datasets.py
```

## Installation

```bash
git clone https://github.com/wzhangwhu/STBSR.git
cd STBSR

conda create -n stbsr python=3.10 -y
conda activate stbsr
pip install -r requirements.txt
```

Install a PyTorch build compatible with the local CUDA driver when GPU
acceleration is required. The runner uses CUDA when available and otherwise
falls back to CPU.

```bash
python -c "import torch, scanpy, anndata; print(torch.__version__, torch.cuda.is_available())"
```

## Data preparation

The repository includes the files required to run the ARTISTA Stage57
telencephalon and DLPFC 151671 notebook examples under
`example_data/ARTISTA_Stage57_telencephalon` and `example_data/151671`. The
remaining benchmark datasets are not redistributed. Dataset registration is
stored in `datasets_31.csv`; update `InputPath` before running the full
benchmark on a new computer.

Each input must contain:

- a gene-expression matrix;
- two-dimensional coordinates in `adata.obsm["spatial"]` or recognized
  coordinate columns in `adata.obs`;
- a ground-truth annotation column matching `GroundTruthKey`.

The manifest also records two dataset-specific parameters:

- `SpatialRadius`: radius used to construct the spatial graph;
- `PlotSpotSize`: marker size used in the spatial-domain figure.

The included benchmark contains 12 DLPFC sections, one human breast cancer
sample, two C1 samples, two human or mouse brain samples, five rat colon tumour
samples, four developing chicken heart samples, three ARTISTA stages, one
Slide-seqV2 mouse embryo sample, and one Stereo-seq mouse brain sample.

## Quick start

Run either bundled example in Jupyter:

```bash
jupyter notebook notebooks/run_stbsr_ARTISTA_Stage57_telencephalon.ipynb
jupyter notebook notebooks/run_stbsr_151671.ipynb
```

Each notebook performs one complete 500-epoch run with seed 1, displays a simple
epoch progress bar, and shows one ground-truth/prediction figure. Neither
notebook writes CSV, h5ad, or PNG outputs.

Validate dependencies, dataset paths, and per-dataset parameters without
training:

```bash
python run_stbsr_31datasets.py --dry-run
```

Run a one-epoch smoke test:

```bash
python run_stbsr_31datasets.py \
  --datasets ARTISTA_Stage57_telencephalon \
  --seeds 1 \
  --epochs 1 \
  --output-root ./outputs/smoke_test
```

Run all 31 datasets with ten random seeds:

```bash
python run_stbsr_31datasets.py \
  --output-root ./outputs/stbsr_31datasets
```

Run selected datasets or seeds:

```bash
python run_stbsr_31datasets.py \
  --datasets HBC 151507 CH_D14 ARTISTA_Stage57_telencephalon \
  --seeds 1 2 3 \
  --epochs 500 \
  --output-root ./outputs/stbsr_subset
```

Use a new or empty output directory for each experiment.

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--output-root` | `outputs/stbsr_31datasets` | Result directory |
| `--manifest` | `datasets_31.csv` | Dataset paths and parameters |
| `--datasets` | All 31 datasets | Dataset names to run |
| `--seeds` | `1 2 ... 10` | Random seeds |
| `--epochs` | `500` | Training epochs per run |
| `--dry-run` | disabled | Validate configuration without training |

## Evaluation

Each run reports ARI, NMI, ACC, purity, entropy, runtime, Moran's score,
boundary intersection-over-union (BIoU), boundary F1 score (BF), and
boundary-band accuracy (BandAcc).

## Outputs

```text
output_root/
|-- all_runs_metrics.csv
|-- summary_mean_std_var_min_max.csv
|-- input_manifest.csv
|-- run_config.json
|-- run_h5ad/
|   `-- <dataset>/
|-- best_h5ad/
|   `-- <dataset>/
`-- spatial_png/
    `-- <dataset>/
```

The runner stores one compact h5ad per run. It contains the ground truth,
predicted domains, pre-clusters, spatial coordinates, STBSR embedding, metrics,
and provenance metadata, but not a duplicated expression matrix. For each
dataset, the highest-ARI run is copied to `best_h5ad`, and exactly one
ground-truth/prediction spatial-domain PNG is written to `spatial_png`.

`all_runs_metrics.csv` contains one row per dataset and seed. The summary CSV
contains the mean, standard deviation, variance, minimum, maximum, and count
for each available metric.

## Data availability

All datasets analysed in this study are publicly available from their original
data providers or associated public repositories. The 31 benchmark datasets and
their access locations are listed below.

| Benchmark datasets | Number | Public source |
|---|---:|---|
| Human dorsolateral prefrontal cortex sections 151507-151510 and 151669-151676 | 12 | [spatialLIBD](https://spatial.libd.org/spatialLIBD/) |
| Human Breast Cancer, Block A, Section 1 | 1 | [10x Genomics](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0) |
| C1_50 and C1_110 control brain | 2 | [STOmicsDB STDS0000201](https://db.cngb.org/stomics/datasets/STDS0000201/summary) |
| Adult Human Brain 1 cerebral cortex | 1 | [10x Genomics](https://www.10xgenomics.com/datasets/adult-human-brain-1-cerebral-cortex-unknown-orientation-stains-anti-gfap-anti-nfh-1-standard-1-1-0); [STOmicsDB STDS0000032](https://db.cngb.org/stomics/datasets/STDS0000032/summary) |
| Mouse Brain Serial Section 1, sagittal-anterior | 1 | [STOmicsDB STDS0000018](https://db.cngb.org/stomics/datasets/STDS0000018/summary) |
| Rat colon tumour samples GSM6505118, GSM6505119, GSM6505121, GSM6505122 and GSM6505120 | 5 | [STOmicsDB STDS0000186](https://db.cngb.org/stomics/datasets/STDS0000186/summary) |
| Developing chicken heart at embryonic days 4, 7, 10 and 14 | 4 | [GEO GSE149457](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149457); [GitHub](https://github.com/madhavmantri/chicken_heart); [Zenodo](https://doi.org/10.5281/zenodo.4517120) |
| ARTISTA axolotl telencephalon at developmental stages 44, 54 and 57 | 3 | [STOmicsDB STDS0000056](https://db.cngb.org/stomics/datasets/STDS0000056/summary) |
| Embryonic day 8.5 mouse embryo Slide-seqV2 | 1 | [GEO GSE197353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE197353); [Figshare E8.5 embryo](https://figshare.com/s/1c29d867bc8b90d754d2) |
| Stereo-seq Mouse_brain sample | 1 | [Figshare Stereo dataset](https://doi.org/10.6084/m9.figshare.28200305) |

The developing chicken heart datasets are associated with Mantri et al. (2021),
and the embryonic mouse Slide-seqV2 dataset is associated with Sampath Kumar et
al. (2023). All datasets must be used in accordance with the access conditions
and terms specified by the original data providers.
