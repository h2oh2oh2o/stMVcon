# stMVcon

**Multi-view Graph Convolutional Network with Cluster-aware Contrastive Learning for Spatial Transcriptomics Domain Identification**

stMVcon is a graph neural network-based method for spatial domain identification in spatial transcriptomics data. It integrates spatial proximity and gene expression similarity through a multi-view graph convolutional framework, and introduces cluster-aware contrastive learning to improve the discriminative ability of learned embeddings for downstream spatial clustering.

This repository provides the implementation of stMVcon and example scripts for reproducing spatial domain identification experiments on 10x Visium spatial transcriptomics datasets, including human dorsolateral prefrontal cortex (DLPFC) and human breast cancer data.

## Overview

Spatial domain identification is an important step in spatial transcriptomics analysis. The goal is to detect tissue regions that show coherent gene expression patterns and spatial organization.

stMVcon models spatial transcriptomics data from two complementary views:

- **Spatial view**: constructed from spatial coordinates to capture local tissue continuity.
- **Feature view**: constructed from gene expression similarity to capture transcriptomic relationships between spots.

The model learns representations from both views using graph convolutional networks and adaptively integrates them. A cluster-aware contrastive learning strategy is further introduced to encourage cross-view consistency while reducing the negative effect of false negative pairs.

The learned embeddings can be used for downstream clustering with methods such as Mclust or KMeans.

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
