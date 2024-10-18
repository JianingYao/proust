import numpy as np
# import pandas as pd
# from sklearn import metrics
from sklearn.decomposition import PCA
# import scanpy as sc
# import matplotlib.pyplot as plt
# from sklearn import metrics
import math


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='profile_gene_img', random_seed=1998):
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


def clustering(adata, n_clusters=7, radius=50, refinement=False, seed=1998, 
                gene_pcs=30, image_pcs=5, gene_weight=None, total_pcs=35, gene_variance=None, img_variance=None):
    """
    Spatial clustering based on the learned representation with optional adjustments
    to gene and image PCs based on user-defined gene weight or cumulative variance.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    refinement : bool, optional
        Refine the predicted labels or not. The default is True.
    gene_pcs : int, optional
        The number of principal components to use for gene data. Default is 30.
    image_pcs : int, optional
        The number of principal components to use for image data. Default is 5.
    gene_weight : float, optional
        User-defined weight for gene data. If provided, gene_pcs and image_pcs will be recalculated.
    total_pcs : int, optional
        Total number of PCs for gene and image data combined. Default is 35.
    gene_variance : float, optional
        Cumulative variance threshold for selecting gene PCs. Overrides gene_pcs if provided.
    img_variance : float, optional
        Cumulative variance threshold for selecting image PCs. Overrides image_pcs if provided.

    Returns
    -------
    None.
    """

    if gene_weight is not None:
        gene_pcs = math.ceil(total_pcs * gene_weight)
        image_pcs = total_pcs - gene_pcs
        print(f"User-defined gene_weight={gene_weight} applied. Adjusted gene_pcs={gene_pcs}, image_pcs={image_pcs}.")

    pca_gene = PCA(random_state=seed)
    embedding_g = pca_gene.fit_transform(adata.obsm['rec_gene'].copy())
    if gene_variance is not None:
        cumulative_variance = np.cumsum(pca_gene.explained_variance_ratio_)
        gene_pcs = np.searchsorted(cumulative_variance, gene_variance) + 1
        print(f"Adjusted gene_pcs based on cumulative variance ({gene_variance}): {gene_pcs}")
    pca_gene = PCA(n_components=gene_pcs, random_state=seed)
    embedding_g = pca_gene.fit_transform(adata.obsm['rec_gene'].copy())

    rec_img = adata.obsm['rec_img'].reshape(adata.obsm['rec_img'].shape[0], -1)
    pca_img = PCA(random_state=seed)
    embedding_i = pca_img.fit_transform(rec_img)
    if img_variance is not None:
        cumulative_variance = np.cumsum(pca_img.explained_variance_ratio_)
        image_pcs = np.searchsorted(cumulative_variance, img_variance) + 1
        print(f"Adjusted image_pcs based on cumulative variance ({img_variance}): {image_pcs}")
    pca_img = PCA(n_components=image_pcs, random_state=seed)
    embedding_i = pca_img.fit_transform(rec_img)

    profile_gene_img = np.concatenate([embedding_g, embedding_i], axis=1)
    adata.obsm['profile_gene_img'] = profile_gene_img

    try:
        adata = mclust_R(adata, used_obsm='profile_gene_img', num_cluster=n_clusters, random_seed=seed)
    except Exception:
        raise RuntimeError(f"Error during Mclust clustering. Please adjust the number of principal components (gene_pcs or image_pcs) and try again.")
    del adata.obsm['profile_gene_img']

    adata.obs['cluster_profile'] = adata.obs['mclust']
    # Optionally refine the labels
    if refinement:
        new_type = refine_label(adata, radius, key='cluster_profile')
        adata.obs['cluster_profile'] = new_type

    return adata.copy()



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