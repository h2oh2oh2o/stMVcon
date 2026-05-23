# stMVcon

**Multi-view Graph Convolutional Network with Cluster-aware Contrastive Learning for Spatial Transcriptomics Domain Identification**

stMVcon is a graph neural network-based method for spatial domain identification in spatial transcriptomics data. It learns spatial-domain-aware representations by integrating two complementary views: a spatial view constructed from tissue coordinates and a feature view constructed from gene expression similarity. A cluster-aware contrastive learning strategy is further introduced to enhance cross-view consistency and improve the clustering-oriented representation learning process.

This repository contains the source code used in our manuscript:

> Multi-view Graph Convolutional Network with Cluster-aware Contrastive Learning for Spatial Transcriptomics Domain Identification

## Overview

Spatial domain identification is an important task in spatial transcriptomics analysis. It aims to identify tissue regions with coherent gene expression profiles and spatial organization. Existing graph-based methods usually construct a spatial graph based on physical proximity and then learn spot embeddings through graph neural networks. However, relying only on spatial proximity may overlook transcriptomically similar spots that are spatially distant, while treating representation learning and clustering as separated stages may limit the clustering quality.

stMVcon addresses these issues by:

- constructing a **spatial view** based on spatial coordinates;
- constructing a **feature view** based on gene expression similarity;
- learning view-specific embeddings using graph convolutional networks;
- adaptively integrating multi-view embeddings;
- introducing **cluster-aware contrastive learning** to improve cross-view consistency and reduce the influence of false negative pairs;
- generating embeddings that can be used for downstream clustering with Mclust or KMeans.

## Repository structure

```text
stMVcon/
├── config/
│   ├── DLPFC.ini
│   └── Human_Breast_Cancer.ini
├── config.py
├── layers.py
├── models.py
├── utils.py
├── test_on_dlpfc.py
├── test_on_hbc.py
├── requirements.txt
└── README.md
```

Description of main files:

| File | Description |
|---|---|
| `config.py` | Reads configuration files and model parameters. |
| `layers.py` | Defines graph convolution layers and neural network components. |
| `models.py` | Implements the main stMVcon model and training process. |
| `utils.py` | Provides utility functions for data processing, graph construction, clustering, evaluation, and visualization. |
| `test_on_dlpfc.py` | Example script for running stMVcon on DLPFC data. |
| `test_on_hbc.py` | Example script for running stMVcon on human breast cancer data. |
| `config/DLPFC.ini` | Configuration file for DLPFC experiments. |
| `config/Human_Breast_Cancer.ini` | Configuration file for human breast cancer experiments. |

## Requirements

The code was developed with Python 3.8. We recommend creating a new conda environment before installation.

```bash
conda create -n stmvcon python=3.8
conda activate stmvcon
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

A recommended `requirements.txt` is shown below:

```text
numpy>=1.24.4
pandas>=1.4.4
scipy>=1.10.1
scikit-learn>=1.3.2
scikit-misc>=0.2.0
matplotlib>=3.7.5
scanpy>=1.9.8
anndata>=0.9.2
h5py>=3.1.0  
faiss-gpu>=1.8.0
fsspec>=2025.3.0
motifcluster>=0.2.3
rpy2>=3.5.15
torch>=2.1.0+cu121
torchvision>=0.16.0+cu121
torch-geometric>=2.5.2
louvain>=0.8.2
```

Notes:

- The package imported as `ot` in the code should be installed as `POT`.
- The package imported as `community` should be installed as `python-louvain`.
- The package imported as `faiss` can be installed as `faiss-cpu` for CPU environments.
- `rpy2` is required when using Mclust through R. Please install R and the R package `mclust` before using Mclust-based clustering.
- The installation of `torch-geometric` may depend on the installed PyTorch and CUDA versions. If `pip install torch-geometric` fails, please install it following the official PyTorch Geometric instructions for your specific environment.

To install the R package `mclust`, run the following command in R:

```r
install.packages("mclust")
```

## Data preparation

Large datasets are not included in this repository due to file size limitations. Please download the corresponding spatial transcriptomics datasets from their original sources and organize them as described below.

### DLPFC data

For the human dorsolateral prefrontal cortex dataset, each tissue section should be stored in an individual folder. The expected directory structure is:

```text
data/
└── DLPFC/
    ├── 151507/
    ├── 151508/
    ├── 151509/
    ├── 151510/
    ├── 151669/
    ├── 151670/
    ├── 151671/
    ├── 151672/
    ├── 151673/
    ├── 151674/
    ├── 151675/
    └── 151676/
```

Each section folder should follow the standard 10x Visium format, for example:

```text
151674/
├── filtered_feature_bc_matrix.h5
├── spatial/
│   ├── tissue_positions_list.csv or tissue_positions.csv
│   ├── scalefactors_json.json
│   ├── tissue_hires_image.png
│   └── tissue_lowres_image.png
└── metadata.tsv
```

The metadata file should contain manual annotation labels for evaluation. For DLPFC data, the annotation column is expected to be:

```text
layer_guess
```

### Human breast cancer data

The human breast cancer data should be organized as:

```text
data/
└── Human_Breast_Cancer/
    ├── filtered_feature_bc_matrix.h5
    ├── spatial/
    │   ├── tissue_positions_list.csv or tissue_positions.csv
    │   ├── scalefactors_json.json
    │   ├── tissue_hires_image.png
    │   └── tissue_lowres_image.png
    └── metadata.tsv
```

The metadata file should contain the ground-truth annotation column used for evaluation.

## Configuration

Model and training parameters are stored in the `.ini` files under the `config/` directory.

Example configuration:

```ini
[Model_Setup]
epochs = 500
lr = 0.001
weight_decay = 1e-4
n_neighbors = 6
n_neighbors1 = 6
preprocess = True
highly_genes = 3000
nhid1 = 256
nhid2 = 64
pca = 50
dropout = 0.0
alpha = 10
beta = 0.01
gamma = 0.5
lamda = 0.5
seed = 3407
refine_radius = 30

[Data_Setting]
fdim = 3000
```

Important parameters:

| Parameter | Description |
|---|---|
| `epochs` | Number of training epochs. |
| `lr` | Learning rate. |
| `weight_decay` | Weight decay for model optimization. |
| `n_neighbors` | Number of neighbors for spatial graph construction. |
| `n_neighbors1` | Number of neighbors for feature graph construction. |
| `highly_genes` | Number of highly variable genes used as input features. |
| `nhid1`, `nhid2` | Hidden dimensions of the neural network. |
| `pca` | Number of principal components used for dimensionality reduction. |
| `alpha`, `beta`, `gamma`, `lamda` | Weights for different loss terms. |
| `seed` | Random seed for reproducibility. |
| `refine_radius` | Radius used for spatial refinement. |

## Usage

### Run stMVcon on DLPFC data

Make sure the DLPFC dataset has been placed under:

```text
./data/DLPFC/
```

Then run:

```bash
python test_on_dlpfc.py
```

By default, the script may run on a selected DLPFC section. To run all 12 DLPFC sections, modify the dataset list in `test_on_dlpfc.py`:

```python
datasets = ['151507', '151508', '151509', '151510',
            '151669', '151670', '151671', '151672',
            '151673', '151674', '151675', '151676']

cluster_num = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]
```

### Run stMVcon on human breast cancer data

Make sure the human breast cancer dataset has been placed under:

```text
./data/Human_Breast_Cancer/
```

Then run:

```bash
python test_on_hbc.py
```

## Output

The output files are saved in the result directory specified in the scripts. Typical outputs include:

```text
result/
├── clustering result figures
├── UMAP visualization figures
├── PAGA visualization figures
└── stMVcon.h5ad
```

The generated `.h5ad` file stores the processed AnnData object, including learned embeddings and predicted cluster labels.

Common fields include:

```python
adata.obsm['emb']    # learned embedding representation
adata.obs['idx']     # predicted cluster labels
```

## Evaluation metrics

The clustering performance can be evaluated using the following metrics:

- **ARI**: Adjusted Rand Index.
- **NMI**: Normalized Mutual Information.
- **HS**: Homogeneity Score.

These metrics are calculated by comparing the predicted cluster labels with manual annotations when ground-truth labels are available.


