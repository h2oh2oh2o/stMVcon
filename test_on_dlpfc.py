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
    datasets = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674',
                '151675', '151676']
    cluster_num = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]
    datasets = ['151674']
    cluster_num = [7]
    results = []
    for i in range(len(datasets)):
        dataset = datasets[i]
        print(dataset)
        config = Config('./config/DLPFC.ini')
        config.class_num = cluster_num[i]
        config.layer = 2
        config.datatype = "10x"
        config.method="mclust"
        path = "D:/单切片空间域识别/数据集/DLPFC/"+ dataset
        # path = "./data/DLPFC/" + dataset
        adata = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        df_meta = pd.read_csv(path + '/metadata.tsv', sep='\t')
        df_meta_layer = df_meta['layer_guess']
        adata.obs['ground_truth'] = df_meta_layer.values
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]

        model = stMVcon(adata=adata, config=config)

        savepath = './result/test/' + dataset + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        labels = model.adata.obs['ground_truth'].copy()
        labels.replace('WM', '0', inplace=True)
        labels.replace('Layer1', '1', inplace=True)
        labels.replace('Layer2', '2', inplace=True)
        labels.replace('Layer3', '3', inplace=True)
        labels.replace('Layer4', '4', inplace=True)
        labels.replace('Layer5', '5', inplace=True)
        labels.replace('Layer6', '6', inplace=True)

        adata, loss_list = model.train(save_reconstruction=True)

        clustering(adata, config.method, config.class_num, config.refine_radius)
        ARI = np.round(metrics.adjusted_rand_score(adata.obs['idx'], labels), 6)
        NMI = np.round(metrics.normalized_mutual_info_score(adata.obs['idx'], labels), 6)
        HS = np.round(metrics.homogeneity_score(adata.obs['idx'], labels), 6)
        print(' ARI = {:.4f}'.format(ARI), ' NMI = {:.4f}'.format(NMI), ' HS = {:.4f}'.format(HS))

        title = 'ARI={:.4f}'.format(ARI) + '  NMI={:.4f}'.format(NMI)+ '  HS={:.4f}'.format(HS)
        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        plt.savefig(savepath + dataset+"_"+title+'.jpg', bbox_inches='tight', dpi=600)


        sc.pp.neighbors(adata, use_rep='emb')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.tl.paga(adata, groups='idx')
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title='', legend_fontoutline=2,
                           show=False)
        plt.savefig(savepath + 'umap.jpg', bbox_inches='tight', dpi=600)

        adata.write(savepath + 'stMVcon.h5ad')
        results.append(ARI)
    print(results)
    print("average is", sum(results) / len(results))
