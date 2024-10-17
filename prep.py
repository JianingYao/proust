import os
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors
import cv2


def resize(spot_img, dim=48):
    new_spot = np.zeros((spot_img.shape[0], dim, dim), dtype='float64')
    for i in range(spot_img.shape[0]):
        new_spot[i] = cv2.resize(spot_img[i], (dim, dim), interpolation=cv2.INTER_AREA)

    return new_spot


def norm_img(adata):
    img_ext = adata.obsm['img_extract']
    max_p = img_ext.max(axis=(0,2,3))
    min_p = img_ext.min(axis=(0,2,3))
    range_p = img_ext.ptp(axis=(0,2,3))
    a = 0
    b = 10.0
    img = np.zeros(img_ext.shape)
    for i in range(img.shape[1]):
        img[:, i, :, :] = a + (img_ext[:, i, :, :] - min_p[i]) * (b - a) / range_p[i]

    adata.obsm['img'] = img


def extract_img(image, adata, r, dim=48):
    x_pixel = adata.obsm['spatial'][:, 1].astype(int)
    y_pixel = adata.obsm['spatial'][:, 0].astype(int)
    img_ext = np.zeros((len(x_pixel), image.shape[0], dim, dim), dtype='float64')
    for i in range(len(x_pixel)):
        max_x = image.shape[1]
        max_y = image.shape[2]
        spot_img = image[:, max(0, x_pixel[i] - r):min(max_x, x_pixel[i] + r),
                   max(0, y_pixel[i] - r):min(max_y, y_pixel[i] + r)]
        img_ext[i, :, :, :] = resize(spot_img, dim)
    adata.obsm['img_extract'] = img_ext
    norm_img(adata)


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


def prefilter_genes(adata, min_counts=None, max_counts=None,min_cells=10, max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)


def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1 = np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)


def prep_gene(adata):
    prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    prefilter_specialgenes(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def permutation(feat):
    ids = np.arange(feat.shape[0])
    ids = np.random.permutation(ids)
    feat_a = feat[ids]
    return feat_a


def norm_gene(gene_feat):
    max_g = gene_feat.max()
    min_g = gene_feat.min()
    range_g = gene_feat.ptp()
    a = 0
    b = 10.0
    gene_norm = a + (gene_feat - min_g) * (b - a) / range_g

    return gene_norm


def initial_feat(adata, random_seed=50):
    adata_Vars = adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        gene_feat = adata_Vars.X.toarray()[:, ]
    else:
        gene_feat = adata_Vars.X[:, ]
    # data augmentation
    fix_seed(random_seed)
    img_feat = adata.obsm['img_feat']
    gene_feat_a = permutation(gene_feat)
    img_feat_a = permutation(img_feat)
    adata.obsm['gene_feat'] = gene_feat
    adata.obsm['gene_feat_a'] = gene_feat_a
    adata.obsm['img_feat_a'] = img_feat_a


def calculate_distance(x):
    """Compute pairwise Euclidean distances."""
    assert isinstance(x, np.ndarray) and x.ndim == 2

    x_square = np.expand_dims(np.einsum('ij,ij->i', x, x), axis=1)
    y_square = x_square.T

    distances = np.dot(x, x.T)
    distances *= -2
    distances += x_square
    distances += y_square

    # Ensure all values are larger than 0
    np.maximum(distances, 0, distances)

    # Ensure that self-distance is set to 0.0
    distances.flat[::distances.shape[0] + 1] = 0.0

    np.sqrt(distances, distances)

    return distances


def construct_interaction(adata, n_neighbors=6):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    # calculate distance matrix
    distance_matrix = calculate_distance(position.astype(np.float64))
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

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj


def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized
