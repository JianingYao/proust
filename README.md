# Proust

## Spatial domain detection using contrastive self-supervised learning for spatial multi-omics technologies

![](Figure_1.jpg)


Recent advances in spatially-resolved single-omics and multi-omics technologies have led to the emergence of computational tools to detect or predict spatial domains. Additionally, histological images and immunofluorescence (IF) staining of proteins and cell types provide multiple perspectives and a more complete understanding of tissue architecture. Here, we introduce Proust, a scalable tool to predict discrete domains using spatial multi-omics data by combining the low-dimensional representation of biological profiles based on graph-based contrastive self-supervised learning. Our method integrates multiple data modalities, such as RNA, protein, and H\&E images, and predicts spatial domains within tissue samples. 
Through the integration of multiple modalities, Proust consistently demonstrates enhanced accuracy in detecting spatial domains, as evidenced across various benchmark datasets and technological platforms.

## Requirements
The following package is required to run proust:
- python==3.10
- torch==1.12.0
- numpy==1.24.4
- tqdm==4.65.0
- scanpy==1.9.3
- scipy==1.11.1
- opencv-python==4.10.0
- scikit-learn==1.2.2
- rpy2==3.5.11

## Tutorial
Please refer to [this tutorial](https://github.com/JianingYao/proust/blob/master/tutorial_IFDLPFC.ipynb) for the step-by-step demonstration on the Visium SPG DLPFC dataset. 


## Citation
Jianing Yao, Jinglun Yu, Brian Caffo, Stephanie C. Page, Keri Martinowich, Stephanie C. Hicks. "Spatial domain detection using contrastive self-supervised learning for spatial multi-omics technologies." bioRxiv, Cold Spring Harbor Laboratory, 2024. [https://www.biorxiv.org/content/10.1101/2024.02.02.578662v1](https://www.biorxiv.org/content/10.1101/2024.02.02.578662v1).


## Contact
Jianing Yao: jyao37@jhmi.edu

Stephanie C. Hicks: shicks19@jhu.edu


