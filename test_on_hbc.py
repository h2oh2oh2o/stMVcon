from __future__ import division
from __future__ import print_function

import ot
import torch.optim as optim
from tqdm import tqdm

from utils import *
import os
import argparse
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = Config('./config/Human_Breast_Cancer.ini')
    config.class_num = 20
    config.layer = 2
    config.datatype = '10x'
    config.method="kmeans"
    path = "./data/Human_Breast_Cancer/"
    adata = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    df_meta = pd.read_csv(path + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['ground_truth']
    adata.obs['ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    model = stMVcon(adata=adata, config=config)

    savepath = './result/Human_Breast_Cancer/result/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)



    labels = model.adata.obs['ground_truth'].copy()
    labels.replace('DCIS/LCIS_1', '0', inplace=True)
    labels.replace('DCIS/LCIS_2', '1', inplace=True)
    labels.replace('DCIS/LCIS_4', '2', inplace=True)
    labels.replace('DCIS/LCIS_5', '3', inplace=True)

    labels.replace('Healthy_1', '4', inplace=True)
    labels.replace('Healthy_2', '5', inplace=True)

    labels.replace('IDC_1', '6', inplace=True)
    labels.replace('IDC_2', '7', inplace=True)
    labels.replace('IDC_3', '8', inplace=True)
    labels.replace('IDC_4', '9', inplace=True)
    labels.replace('IDC_5', '10', inplace=True)
    labels.replace('IDC_6', '11', inplace=True)
    labels.replace('IDC_7', '12', inplace=True)
    labels.replace('IDC_8', '13', inplace=True)

    labels.replace('Tumor_edge_1', '14', inplace=True)
    labels.replace('Tumor_edge_2', '15', inplace=True)
    labels.replace('Tumor_edge_3', '16', inplace=True)
    labels.replace('Tumor_edge_4', '17', inplace=True)
    labels.replace('Tumor_edge_5', '18', inplace=True)
    labels.replace('Tumor_edge_6', '19', inplace=True)

    adata, loss_list = model.train(True)

    clustering(adata, config.method, config.class_num, config.refine_radius)
    ari_refined = np.round(metrics.adjusted_rand_score(adata.obs['idx'], labels), 6)
    NMI = np.round(metrics.normalized_mutual_info_score(adata.obs['idx'], labels), 6)
    HS = np.round(metrics.homogeneity_score(adata.obs['idx'], labels), 6)
    print(' ARI = {:.6f}'.format(ari_refined), ' NMI = {:.6f}'.format(NMI), ' HS = {:.6f}'.format(HS))

    plt.rcParams["figure.figsize"] = (4, 4)
    title = 'ARI={:.4f}'.format(ari_refined) + '   NMI={:.4f}'.format(NMI)
    sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
    plt.savefig(savepath + 'stMVcon.jpg', bbox_inches='tight', dpi=600)

    plt.rcParams["figure.figsize"] = (4, 4)
    title = "Manual annotation"
    sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
    plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=600)
    adata.layers['X'] = adata.X
    adata.write(savepath + 'stMVcon.h5ad')
