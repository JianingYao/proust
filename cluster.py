import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn import metrics


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='profile', random_seed=12321):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

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
    return adata.copy()


def clustering(adata, dataset, n_clusters=7, npc_gene=30, npc_img=5, radius=50, refinement=False, seed=12321):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    threshold : float, optional
        Cutoff for selecting the final labels. For 10X Visium data, the model is trained twice,
        i.e., with and without penalty terms. As a result,  two clustering label results corresponding to the
        two-time training are generated. The final clustering label is determined by Silhouette score.
        The default is 0.06.
    refinement : bool, optional
        Refine the predicted labels or not. The default is True.

    Returns
    -------
    None.

    """

    pca_gene = PCA(n_components=npc_gene, random_state=seed)
    pca_img = PCA(n_components=npc_img, random_state=seed)

    embedding_g = pca_gene.fit_transform(adata.obsm['rec_gene'].copy())

    rec_img = adata.obsm['rec_img']
    rec_img = adata.obsm['rec_img'].reshape(rec_img.shape[0], -1)
    embedding_i = pca_img.fit_transform(rec_img)

    profile_img_pca = np.concatenate([embedding_g, embedding_i], axis=1)
    adata.obsm['profile'] = profile_img_pca

    adata_profile = mclust_R(adata, used_obsm='profile', num_cluster=n_clusters)
    adata_profile.obs['label'] = adata_profile.obs['mclust']
    if refinement:
        new_type = refine_label(adata_profile, radius, key='label')
        adata.obs['label'] = new_type

    return adata_profile.copy()


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # read distance
    if 'distance_matrix' not in adata.obsm.keys():
        raise ValueError("Distance matrix is not existed!")
    distance = adata.obsm['distance_matrix'].copy()

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

    return new_type

