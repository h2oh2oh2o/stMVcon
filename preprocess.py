import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix


def normalize(adata, datatype=None, highly_genes=3000):
    print("start select HVGs")
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def get_feature(adata, preprocess='True'):
    if not preprocess:
        adata_Vars = adata
    else:
        adata_Vars = adata[:, adata.var['highly_variable']]

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]
    adata.obsm['feat'] = feat


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    return adj_normalized
