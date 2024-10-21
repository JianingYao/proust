import torch
# import time
# import random
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
# from matplotlib import pyplot as plt
# from scipy.sparse.csc import csc_matrix
# from scipy.sparse.csr import csr_matrix
# import pandas as pd
import math

from proust.prep import *
from proust.nnModels import *

def Img_learn(adata, image, lr=0.001, epochs=1000, device = 'mps', random_seed=1998):
    slide = list(adata.uns['spatial'].keys())[0]
    spot_dm = adata.uns['spatial'][slide]['scalefactors']['spot_diameter_fullres']
    r = math.ceil(spot_dm / 2)
    print("Image dimension r: ", r)
    extract_img(image, adata, r)

    # construct conv model for image feature extraction
    n_channels = image.shape[0]
    img = torch.FloatTensor(adata.obsm['img'].copy()).to(device)
    torch.manual_seed(random_seed)
    model_img = Img_cov(n_channels).to(device)
    optim_img = torch.optim.Adam(model_img.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        model_img.train()
        optim_img.zero_grad()
        img_rec, _ = model_img(img)
        loss = F.mse_loss(img_rec, img)
        loss.backward()
        optim_img.step()

    with torch.no_grad():
        model_img.eval()
        _, img_feat_final = model_img(img)

    adata.obsm['img_feat'] = img_feat_final.detach().cpu().numpy().reshape(img_feat_final.shape[0], img_feat_final.shape[1], -1)


class Hybrid_train():
    def __init__(self, adata, device='mps', learning_rate=0.001, weight_decay=0.00, epochs=600, dim_output=64, alpha=10, beta=1):
        self.adata = adata
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta

        self.gene = torch.FloatTensor(adata.obsm['gene_feat'].copy()).to(self.device)
        self.gene_a = torch.FloatTensor(adata.obsm['gene_feat_a'].copy()).to(self.device)
        self.img = torch.FloatTensor(adata.obsm['img_feat'].copy()).to(self.device)
        self.img_a = torch.FloatTensor(adata.obsm['img_feat_a'].copy()).to(self.device)

        self.label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(self.device)
        self.adj = adata.obsm['adj']

        self.adj = preprocess_adj(self.adj)
        self.adj = torch.FloatTensor(self.adj).to(self.device)
        self.graph_neigh = torch.FloatTensor(adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)

        self.dim_g_input = self.gene.shape[1]
        self.dim_i_input = self.img.shape[2]
        self.dim_output_g = dim_output
        self.img_n = self.img.shape[1]

        self.loss_CSL = nn.BCEWithLogitsLoss()

    def loss_function(self, pos, neg):
        loss_sl_1 = 0
        loss_sl_2 = 0
        for i in range(pos.shape[1]):
            loss_sl_1 += self.loss_CSL(pos[:, i, :], self.label_CSL)
            loss_sl_2 += self.loss_CSL(neg[:, i, :], self.label_CSL)

        loss_sl_1 = loss_sl_1 / pos.shape[1]
        loss_sl_2 = loss_sl_2 / pos.shape[1]

        return loss_sl_1, loss_sl_2

    def train_gene(self):
        # Reconstruct gene expression
        self.model_g = autoencoder_g(self.dim_g_input, self.dim_output_g, self.graph_neigh, self.device).to(self.device)
        self.optim_g = torch.optim.Adam(self.model_g.parameters(), self.learning_rate, weight_decay=self.weight_decay)

        print('Begin to train spatial transcriptomics data')
        self.model_g.train()

        for epoch in tqdm(range(self.epochs)):
            self.model_g.train()
            self.gene_a = permutation(self.gene).to(self.device)
            self.hid_gene, self.hid_gene_a, self.rec_gene, pos_g, neg_g = self.model_g(self.gene, self.gene_a, self.adj)

            loss_sl_1_g = self.loss_CSL(pos_g, self.label_CSL)
            loss_sl_2_g = self.loss_CSL(neg_g, self.label_CSL)
            loss_coder_g = F.mse_loss(self.gene, self.rec_gene)
            loss_g = self.alpha * loss_coder_g + self.beta * (loss_sl_1_g + loss_sl_2_g)

            self.optim_g.zero_grad()
            loss_g.backward()
            self.optim_g.step()

        print("Optimization finished for ST data!")

        with torch.no_grad():
            self.model_g.eval()
            self.emb_gene = self.model_g(self.gene, self.gene_a, self.adj)[0].detach().cpu().numpy()
            self.rec_gene = self.model_g(self.gene, self.gene_a, self.adj)[2].detach().cpu().numpy()

        return self.emb_gene, self.rec_gene


    def train_img(self):
        # Obtain protein profile
        self.model_i = autoencoder_i(self.dim_i_input, self.graph_neigh, self.img_n, self.device).to(self.device)
        self.optim_i = torch.optim.Adam(self.model_i.parameters(), self.learning_rate, weight_decay=self.weight_decay)

        print('Begin to train protein information')
        self.model_i.train()

        for epoch in tqdm(range(self.epochs)):
            self.model_i.train()
            self.img_a = permutation(self.img).to(self.device)
            self.img_score, self.rec_img, pos_i, neg_i = self.model_i(self.img, self.img_a, self.adj)

            loss_sl_1_i, loss_sl_2_i = self.loss_function(pos_i, neg_i)
            loss_coder_i = F.mse_loss(self.img, self.rec_img)
            loss_i = self.alpha * loss_coder_i + self.beta * (loss_sl_1_i + loss_sl_2_i)

            self.optim_i.zero_grad()
            loss_i.backward()
            self.optim_i.step()

        print("Optimization finished for protein profile!")

        with torch.no_grad():
            self.model_i.eval()
            self.img_score = self.model_i(self.img, self.img_a, self.adj)[0].detach().cpu().numpy()
            self.rec_img = self.model_i(self.img, self.img_a, self.adj)[1].detach().cpu().numpy()

        return self.img_score, self.rec_img


class proust():
    def __init__(self, adata, random_seed=50, device='mps'):
        self.adata = adata
        self.random_seed = random_seed
        self.device = device

        self.img = adata.obsm['img_feat']
        fix_seed(self.random_seed)
        prep_gene(self.adata)
        initial_feat(self.adata, self.random_seed)

        construct_interaction(self.adata)
        add_contrastive_label(self.adata)


    def train(self):
        model = Hybrid_train(self.adata, device=self.device)
        _, rec_gene = model.train_gene()
        _, rec_img = model.train_img()
        self.adata.obsm['rec_gene'] = rec_gene
        self.adata.obsm['rec_img'] = rec_img
        del self.adata.obsm['gene_feat']
        del self.adata.obsm['gene_feat_a']
        del self.adata.obsm['img_feat']
        del self.adata.obsm['img_feat_a']
        del self.adata.obsm['img_extract']
        del self.adata.obsm['img']
        del self.adata.obsm['label_CSL']
        del self.adata.obsm['adj']
        del self.adata.obsm['graph_neigh']
        
        return self.adata