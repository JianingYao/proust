import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
import cv2
import imageio
import numpy as np
from numpy import newaxis
import sys

from Train import *
from cluster import *
from prep import *


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# if using Macbook GPU chip
device = torch.device("mps")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'

dir = "/Users/jianingyao/Desktop/Research/Biostatistics_JHU/PhD/Data"

n_clusters = 7
dataset = '8667' 

print("################################ sample-" + str(dataset) + " #####################################")
file_fold = dir + '/Visium-DLPFC/' + str(dataset)
adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()

image = imageio.volread(dir + "/Visium-DLPFC/" + str(dataset) + "/" + str(dataset) + ".tif")
image = image[1:6, :, :]

# Extract img features
Img_learn(adata, image, device=device)
print("Finish extracting image features!")

model = proST(adata, device=device)
adata = model.train()
adata.write_h5ad(str(dataset) + "_results.h5ad")
adata = sc.read(str(dataset) + "_results.h5ad")

# radius of the number of neighbors during refinement
radius = 10
adata = clustering(adata, dataset, n_clusters=n_clusters, radius=radius, refinement=True)

adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
adata.obsm['spatial'] = adata.obsm['spatial'].astype(int)

dpi = 500
fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
sc.pl.embedding(adata,
                        basis="spatial",
                        color="label",
                        size=40,
                        show=False,
                        ax=ax)
plt.savefig(str(dataset) + '_refPred.png',
            dpi=dpi, bbox_inches='tight')
plt.close()

