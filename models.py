from math import ceil

import faiss
import fsspec.asyn
import torch
from sklearn import metrics
import torch.optim as optim
from tqdm import tqdm
import motifcluster.motifadjacency as motif
from utils import *
from preprocess import *


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class UncertaintyMLP(nn.Module):
    def __init__(self, input_dim, n_clusters):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2 * n_clusters)  # 输出均值和方差

    def forward(self, x):
        x = F.relu(x)
        out = self.fc(x)
        mu, logvar = out.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        samples = mu + std * torch.randn_like(std)
        return samples


class Encoder2(Module):
    def __init__(self, nfeat, nhid, out, class_num, dropout=0.0, act=F.relu_):
        super(Encoder2, self).__init__()

        self.dropout = dropout
        self.act = act

        self.gcn1_en = GraphConvolution(nfeat, nhid)
        self.gcn2_en = GraphConvolution(nhid, out)

        self.linear1_de = nn.Linear(out, nhid)
        self.linear2_de = nn.Linear(nhid, nfeat)
        self.softAssign = UncertaintyMLP(out, class_num)
        self.att = Attention(out)

    def forward(self, x, adj, adj1):
        x = F.dropout(x, self.dropout, self.training)

        # encoder
        x1 = self.gcn1_en(x, adj)
        x1 = self.act(x1)
        x1 = self.gcn2_en(x1, adj)

        x2 = self.gcn1_en(x, adj1)
        x2 = self.act(x2)
        x2 = self.gcn2_en(x2, adj1)

        emb = torch.stack([x1, x2], dim=1)
        emb, _ = self.att(emb)

        label = F.softmax(self.softAssign(emb), dim=-1)

        x_ = self.linear1_de(emb)
        x_ = self.act(x_)
        x_ = self.linear2_de(x_)

        return x1, x2, emb, x_, label


class Encoder1(Module):
    def __init__(self, nfeat, out, class_num, dropout=0.0, act=F.relu_):
        super(Encoder1, self).__init__()

        self.dropout = dropout
        self.act = act

        self.gcn1_en = GraphConvolution(nfeat, out)

        self.linear1_de = nn.Linear(out, nfeat)
        self.softAssign = UncertaintyMLP(out, class_num)
        self.att = Attention(out)

    def forward(self, x, adj, adj1):
        x = F.dropout(x, self.dropout, self.training)

        # encoder
        x1 = self.gcn1_en(x, adj)

        x2 = self.gcn1_en(x, adj1)

        emb = torch.stack([x1, x2], dim=1)
        emb, _ = self.att(emb)

        label = F.softmax(self.softAssign(emb), dim=-1)

        x_ = self.linear1_de(emb)

        return x1, x2, emb, x_, label


class stMVcon(nn.Module):
    def __init__(self, adata, config):
        super(stMVcon, self).__init__()
        self.adata = adata.copy()
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        fix_seed(config.seed)
        if config.preprocess:
            normalize(self.adata, self.config.datatype, self.config.highly_genes)

        get_feature(self.adata, self.config.preprocess)

        if 'sadj' not in adata.obsm.keys():
            if config.radius:
                construct_interaction_spatial1(self.adata, config.radius)
            else:
                construct_interaction_spatial(self.adata, self.config.n_neighbors)

        if 'fadj' not in adata.obsm.keys():
            features_construct_graph(self.adata, k=self.config.n_neighbors1, pca=config.pca)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.sadj = self.adata.obsm['sadj']  # .toarray()
        self.fadj = self.adata.obsm['fadj']  # .toarray()

        self.input_dim = self.config.fdim
        self.hidden_dim = self.config.nhid1
        self.out_dim = self.config.nhid2

        self.normalized_sadj = preprocess_adj(self.sadj)  # 对邻接矩阵标准化
        self.normalized_sadj = torch.FloatTensor(self.normalized_sadj).to(self.device)
        Wm = motif.build_motif_adjacency_matrix(self.fadj, "M3", "func", "mean")
        self.normalized_fadj = preprocess_adj(Wm.toarray())  # 对邻接矩阵标准化
        self.normalized_fadj = torch.FloatTensor(self.normalized_fadj).to(self.device)

        self.sadj = torch.FloatTensor(self.sadj).to(self.device)
        self.fadj = torch.FloatTensor(self.fadj).to(self.device)

        if config.layer == 2:
            self.model = Encoder2(self.input_dim, self.hidden_dim, self.out_dim, config.class_num, config.dropout).to(
                self.device)
        else:
            self.model = Encoder1(self.input_dim, self.out_dim, config.class_num, config.dropout).to(
                self.device)

    def train(self, save_reconstruction=False):

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        loss=0
        for epoch in tqdm(range(self.config.epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            emb1, emb2, hidden_emb, rec, label = self.model(self.features, self.normalized_sadj,
                                                            self.normalized_fadj)
            self.bceloss = nn.BCEWithLogitsLoss()
            rec_loss1 = F.mse_loss(self.features, rec)
            rec_loss2 = self.bceloss(torch.matmul(hidden_emb, hidden_emb.t()), self.sadj)
            con_loss1 = feat_regularization(label, hidden_emb)

            hard_label = F.gumbel_softmax(label.detach(), hard=True, dim=1)
            hard_label = torch.matmul(hard_label, hard_label.t())
            self_mask = torch.eye(emb1.size(0), dtype=bool, device=hidden_emb.device)
            pos_mask = (hard_label>0) & (~self_mask)
            target_matrix = torch.eye(emb1.size(0), device=hidden_emb.device)
            confidence = torch.sigmoid(torch.matmul(F.normalize(hidden_emb, dim=1), F.normalize(hidden_emb, dim=1).t())[pos_mask].detach().mean())
            target_matrix[pos_mask] = confidence
            con_loss2 = F.mse_loss(torch.matmul(F.normalize(emb1, dim=1), F.normalize(emb2, dim=1).t()), target_matrix)

            total_loss = self.config.alpha * rec_loss1 + self.config.beta * rec_loss2+self.config.gamma*con_loss1 + self.config.lamda * con_loss2
            loss = rec_loss1+rec_loss2+con_loss1+con_loss2
            loss = loss.detach().cpu().numpy()
            total_loss.backward()
            self.optimizer.step()
        self.adata.obsm['emb'] = hidden_emb.cpu().detach().numpy()
        if save_reconstruction:
            self.adata = self.adata[:, self.adata.var['highly_variable']]
            X = pd.DataFrame(self.adata.X.toarray()[:, ], index=self.adata.obs.index, columns=self.adata.var.index)
            rec = pd.DataFrame(rec.cpu().detach().numpy(), index=X.index, columns=X.columns)
            rec[rec < 0] = 0
            self.adata.layers['rec'] = rec.values
        return self.adata, loss


def feat_regularization(label, emb, temp=1):
    feat = torch.mm(label.t(), emb.detach())
    mask = torch.eye(feat.size(0), dtype=torch.bool, device=feat.device)
    feat = F.normalize(feat, dim=-1)
    sim = torch.sigmoid(torch.matmul(feat, feat.t()) / temp)

    k = label.size(1)
    s = label.unsqueeze(0) if label.dim() == 2 else label
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    loss = torch.log(sim[~mask]).mean() + 0.1 * ortho_loss
    return loss
