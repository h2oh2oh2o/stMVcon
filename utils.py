import os
import random
from torch.backends import cudnn
import ot
import scipy.sparse as sp
import sklearn
import torch
import networkx as nx
from sklearn.cluster import KMeans
import community as community_louvain
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import config

from layers import *
import torch
from utils import *
from torch.nn.parameter import Parameter
from torch_geometric.nn import dense_mincut_pool
from sklearn.metrics import silhouette_score
import torch
import torch.nn.functional as F

EPS = 1e-15

import torch
from torch import Tensor


def regularization_loss(emb, graph_nei, graph_neg, sadj):
    sadj = sadj.to_dense()
    emb = readout_function(emb, sadj)
    mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()
    # mat = pd.DataFrame(mat.cpu().detach().numpy()).values

    # graph_neg = torch.ones(graph_nei.shape) - graph_nei

    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean()
    pair_loss = -(neigh_loss + neg_loss) / 2
    return pair_loss


def consistency_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    return torch.mean((cov1 - cov2) ** 2)


def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def cosine_similarity_matrix(X):
    """
    计算两个矩阵行向量之间的余弦相似度

    参数:
    X (np.ndarray): 形状为 (m, d) 的矩阵

    返回:
    np.ndarray: 形状为 (m, m) 的余弦相似度矩阵
    """
    # 计算点积：X 的每行与 Y 的每行的点积
    dot_product = X @ X.T  # 形状为 (m, n)

    # 计算范数
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)  # 形状为 (m, 1)

    # 计算余弦相似度
    similarity = dot_product / (X_norm @ X_norm.T + 1e-8)  # 加小常数避免除零

    return similarity


def kld(target, pred):
    return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))


def construct_interaction_spatial1(adata, radius=150):
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A = np.zeros((coor.shape[0], coor.shape[0]))

    # print("coor:", coor)
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]] = 1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs))

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    adata.obsm['sadj'] = sadj.toarray()


def calculate_dist(coor):
    coor1 = coor.unsqueeze(1)  # 形状变为 (n, 1, 2)
    coor2 = coor.unsqueeze(0)  # 形状变为 (1, n, 2)

    # 计算差值的平方
    diff = (coor1 - coor2) ** 2

    # 对差值的平方求和并开方
    distances = torch.sqrt(torch.sum(diff, dim=-1))
    return distances


def celltype_dec_construct_graph(celltype_dec, k=5, pca=None, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    if pca is not None:
        features = dopca(celltype_dec, dim=pca)
    # print("k: ", k)
    # print("features_construct_graph features", features.shape)
    A = kneighbors_graph(celltype_dec, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    # index = np.argwhere(A > 0)
    # np.savetxt('./GAE+AE+mincut+AE/fadj.csv', index, delimiter=',')
    cadj = sp.coo_matrix(A, dtype=np.float32)
    cadj = cadj + cadj.T.multiply(cadj.T > cadj) - cadj.multiply(cadj.T > cadj)
    # nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return cadj  # , nfadj


def features_construct_graph(adata, k=15, pca=0, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    features = adata.X
    if pca :
        if isinstance(features, np.ndarray):
            features = dopca(features, dim=pca)
        else:
            features = dopca(features.toarray(), dim=pca)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    fadj = fadj.toarray()
    fadj = fadj + np.eye(fadj.shape[0])
    adata.obsm['fadj'] = fadj#.toarray()

def search_res(adata, n_clusters, used_obsm='emb', method='louvain', start=2.0, step=0.01, max_run=50, prior=True):
    sc.pp.neighbors(adata, use_rep=used_obsm)
    res=start
    print("Start at res = ", res, "step = ", step)
    if method == 'leiden':
        sc.tl.leiden(adata, random_state=2023, resolution=res)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        print('resolution={}, cluster number={}'.format(res, count_unique))
        run = 1
        while count_unique != n_clusters:
            run += 1
            old_sign = 1 if (count_unique < n_clusters) else -1
            res = res + step * old_sign
            res = round(res, 4)
            sc.tl.leiden(adata, random_state=2023, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
            new_sign = 1 if (count_unique < n_clusters) else -1
            if new_sign != old_sign:
                step = round(step / 2, 4)
                print("Step changed to", step)
            if run >= max_run:
                print("The maximum number of searches has been reached! Exact resolution not found!")
                break
    elif method == 'louvain':
        sc.tl.louvain(adata, random_state=2023, resolution=res)
        count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
        print('resolution={}, cluster number={}'.format(res, count_unique))
        run = 1
        while count_unique != n_clusters:
            run += 1
            old_sign = 1 if (count_unique < n_clusters) else -1
            res = res + step * old_sign
            res = round(res, 4)
            sc.tl.louvain(adata, random_state=2023, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
            new_sign = 1 if (count_unique < n_clusters) else -1
            if new_sign != old_sign:
                step = step / 2
                print("Step changed to", step)
            if run >= max_run:
                print("The maximum number of searches has been reached! Exact resolution not found!")
                break

    print("recommended res = ", str(res))
    return res

def clustering(adata, method, class_num, refine_radius, used_obsm='emb', start=2.0, step=0.01, max_run=50):
    if method == "kmeans":
        emb = pd.DataFrame(adata.obsm[used_obsm]).fillna(0).values
        kmeans = KMeans(n_clusters=class_num, n_init="auto").fit(emb)
        idx = kmeans.labels_
        adata.obs['idx'] = idx.astype(str)
    elif method == "mclust":
        adata = mclust_R(adata, used_obsm=used_obsm, num_cluster=class_num)
        adata.obs['idx'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, class_num, used_obsm=used_obsm, method=method, start=start, step=step, max_run=max_run)
        sc.tl.leiden(adata, random_state=2023, resolution=res)
        adata.obs['idx'] = adata.obs['leiden'].astype('category')
    elif method == 'louvain':
        res = search_res(adata, class_num, used_obsm=used_obsm, method=method, start=start, step=step, max_run=max_run)
        sc.tl.louvain(adata, random_state=2023, resolution=res)
        adata.obs['idx'] = adata.obs['louvain'].astype('category')
    # 结果微调
    if refine_radius:
        print("--------------------------------begin refined---------------------------------")
        new_type = refine_label(adata, refine_radius, key='idx')
        adata.obs['idx'] = new_type
        print("--------------------------------end---------------------------------")

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    # -*- coding : utf-8-*-
    # coding:unicode_escape

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def get_adj(data, pca=None, k=25, mode="connectivity", metric="cosine"):
    if pca is not None:
        data = dopca(data, dim=pca)
        data = data.reshape(-1, 1)
    A = kneighbors_graph(data, k, mode=mode, metric=metric, include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    # S = cosine_similarity(data)
    return adj, adj_n  # , S


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def dopca(data, dim=50):
    return PCA(n_components=dim).fit_transform(data)


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


class louvain:
    def __init__(self, level):
        self.level = level
        return

    def updateLabels(self, level):
        # Louvain algorithm labels community at different level (with dendrogram).
        # Here we want the community labels at a given level.
        level = int((len(self.dendrogram) - 1) * level)
        partition = community_louvain.partition_at_level(self.dendrogram, level)
        # Convert dictionary to numpy array
        self.labels = np.array(list(partition.values()))
        return

    def update(self, inputs, adj_mat=None):
        """Return the partition of the nodes at the given level.

        A dendrogram is a tree and each level is a partition of the graph nodes.
        Level 0 is the first partition, which contains the smallest communities,
        and the best is len(dendrogram) - 1.
        Higher the level is, bigger the communities are.
        """
        self.graph = nx.from_numpy_matrix(adj_mat)
        self.dendrogram = community_louvain.generate_dendrogram(self.graph)
        self.updateLabels(self.level)
        self.centroids = computeCentroids(inputs, self.labels)
        return


def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([np.mean(data[labels == i], axis=0) for i in range(n_clusters)])


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx



class Colors():
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#633194",
        "#8c564b",
        "#F73BAD",
        "#ad494a",
        "#F6E800",
        "#01F7F7",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#dbdb8d",
        "#9edae5",
        "#8c6d31"]


def res_search_fixed_clus(cluster_type, adata, fixed_clus_count, increment=0.01):
    '''
                arg1(adata)[AnnData matrix]
                arg2(fixed_clus_count)[int]

                return:
                    resolution[int]
            '''
    if cluster_type == 'leiden':
        for res in sorted(list(np.arange(0.14, 2.5, increment))):  # , reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            # print(res,' ' , count_unique_leiden)
            if count_unique_leiden == fixed_clus_count:
                cluster_labels = np.array(adata.obs['leiden'])
                flag = 0
                break
            if count_unique_leiden > fixed_clus_count:
                cluster_labels = np.array(adata.obs['leiden'])
                flag = 1
                break
    elif cluster_type == 'louvain':
        for res in sorted(list(np.arange(0.14, 2.5, increment))):  # , reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            # print(res,' ' , count_unique_louvain)
            if count_unique_louvain == fixed_clus_count:
                cluster_labels = np.array(adata.obs['louvain'])
                flag = 0
                break
            if count_unique_louvain > fixed_clus_count:
                cluster_labels = np.array(adata.obs['louvain'])
                flag = 1
                break
    return cluster_labels, flag


def PCA_process(X, nps):
    from sklearn.decomposition import PCA
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)  # 等价于pca.fit(X) pca.transform(X)
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum())
    return X_PC


def construct_interaction_spatial(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']

    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    # adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)
    adj = adj+np.eye(adj.shape[0])
    adata.obsm['sadj'] = adj


def set_seed(config):
    import random
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)

    if not config.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def build_adjacency_matrix(idx):
    # 将idx转换为张量（如果不是）并确保是整数类型
    if not isinstance(idx, torch.Tensor):
        idx = torch.tensor(idx)

    # 确保idx是整数类型
    if idx.dtype != torch.long:
        idx = idx.long()  # 转换为long类型（整数类型）

    encoder = OneHotEncoder(sparse_output=False)
    label = torch.FloatTensor(encoder.fit_transform(idx.reshape(-1, 1))).cuda()
    adj_matrix = torch.matmul(label, label.t())
    return adj_matrix
    # return normalize_sparse_matrix_torch(adj_matrix.to_sparse())


def normalize_sparse_matrix_torch(sparse_mx):
    """
    PyTorch版本的对称归一化稀疏矩阵: D^(-1/2) * mx * D^(-1/2)

    参数:
        sparse_mx: PyTorch稀疏矩阵 (torch.sparse.FloatTensor)

    返回:
        归一化后的稀疏矩阵
    """
    # 计算每行的和（度矩阵的对角线元素）
    rowsum = torch.sparse.sum(sparse_mx, dim=1).to_dense()

    # 计算D^(-1/2)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.

    # 创建对角矩阵D^(-1/2)
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)

    # 计算 D^(-1/2) * mx
    mx_norm = torch.sparse.mm(sparse_mx, r_mat_inv_sqrt)

    # 计算 (D^(-1/2) * mx) * D^(-1/2)
    mx_norm = torch.mm(mx_norm, r_mat_inv_sqrt)

    return mx_norm



def normalize_sparse_matrix_torch(sparse_mx):
    """
    PyTorch版本的对称归一化稀疏矩阵: D^(-1/2) * mx * D^(-1/2)

    参数:
        sparse_mx: PyTorch稀疏矩阵 (torch.sparse.FloatTensor)

    返回:
        归一化后的稀疏矩阵
    """
    # 计算每行的和（度矩阵的对角线元素）
    rowsum = torch.sparse.sum(sparse_mx, dim=1).to_dense()

    # 计算D^(-1/2)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.

    # 创建对角矩阵D^(-1/2)
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)

    # 计算 D^(-1/2) * mx
    mx_norm = torch.sparse.mm(sparse_mx, r_mat_inv_sqrt)

    # 计算 (D^(-1/2) * mx) * D^(-1/2)
    mx_norm = torch.mm(mx_norm, r_mat_inv_sqrt)

    return mx_norm


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type


def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0:num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()
