# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from proust.prep import *


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)


class Img_cov(nn.Module):
    def __init__(self, n_channels, kernel_size=5):
        super(Img_cov, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                               groups=n_channels)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                               groups=n_channels)
        self.pool = nn.AvgPool2d(2, 2)
        self.iconv1 = nn.ConvTranspose2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size + 1,
                                         stride=2, groups=n_channels)
        self.iconv2 = nn.ConvTranspose2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size + 1,
                                         stride=2, groups=n_channels)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def decode(self, x):
        x = F.relu(self.iconv1(x))
        x = self.iconv2(x)
        return x

    def forward(self, x):
        lat = self.encode(x)
        rec = self.decode(lat)
        return rec, lat


class autoencoder_g(nn.Module):
    def __init__(self, in_features, out_features, graph_neigh, device, dropout=0.0, act=F.relu):
        super(autoencoder_g, self).__init__()
        self.in_features = in_features
        self.out_features = out_features # 32/64
        self.graph_neigh = graph_neigh
        self.device = device
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)
        hidden_emb = z

        recon = torch.mm(z, self.weight2)
        recon = torch.spmm(adj, recon)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        hidden_emb_a = z_a

        emb = self.act(z)
        emb_a = self.act(z_a)

        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        pos = self.disc(g, emb, emb_a)
        neg = self.disc(g_a, emb_a, emb)

        return hidden_emb, hidden_emb_a, recon, pos, neg


class autoencoder_i(nn.Module):
    def __init__(self, in_features, graph_neigh, img_n, device, out_features=8, dropout=0.0, act=F.relu):
        super(autoencoder_i, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.img_n = img_n
        self.device = device
        self.dropout = dropout
        self.act = act

        self.en_weight1 = Parameter(torch.FloatTensor(self.in_features, self.img_n, self.out_features))
        self.de_weight1 = Parameter(torch.FloatTensor(self.out_features, self.img_n, self.in_features))
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        for i in range(self.img_n):
            torch.nn.init.xavier_uniform_(self.en_weight1[:, i, :])
            torch.nn.init.xavier_uniform_(self.de_weight1[:, i, :])

    def channel_learning(self, feat, feat_a, adj, channel):
        z = F.dropout(feat, self.dropout, self.training)
        # layer 1
        z = torch.mm(z, self.en_weight1[:, channel, :])
        z = torch.spmm(adj, z)
        hidden_emb = z

        recon = torch.mm(z, self.de_weight1[:, channel, :])
        recon = torch.spmm(adj, recon)

        # corrupted data
        z_a = F.dropout(feat_a, self.dropout, self.training)
        # layer 1
        z_a = torch.mm(z_a, self.en_weight1[:, channel, :])
        z_a = torch.spmm(adj, z_a)
        hidden_emb_a = z_a

        # contrastive learning
        emb = self.act(z)
        emb_a = self.act(z_a)
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)
        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)
        pos = self.disc(g, emb, emb_a)
        neg = self.disc(g_a, emb_a, emb)

        return hidden_emb, hidden_emb_a, recon, pos, neg

    def forward(self, img, img_a, adj):
        hid_emb_channels = torch.FloatTensor(img.shape[0], self.img_n, self.out_features).to(self.device)
        hid_emb_a_channels = torch.FloatTensor(img.shape[0], self.img_n, self.out_features).to(self.device)
        recon_channels = torch.FloatTensor(img.shape[0], self.img_n, img.shape[2]).to(self.device)
        pos_channels = torch.FloatTensor(img.shape[0], self.img_n, 2).to(self.device)
        neg_channels = torch.FloatTensor(img.shape[0], self.img_n, 2).to(self.device)

        for i in range(self.img_n):
            hid_emb_single, hid_emb_a_single, recon_single, pos_single, neg_single = self.channel_learning(img[:, i, :], img_a[:, i, :], adj, i)
            hid_emb_channels[:, i, :] = hid_emb_single
            hid_emb_a_channels[:, i, :] = hid_emb_a_single
            recon_channels[:, i, :] = recon_single
            pos_channels[:, i, :] = pos_single
            neg_channels[:, i, :] = neg_single

        score = hid_emb_channels.reshape(img.shape[0], -1)

        return score, recon_channels, pos_channels, neg_channels