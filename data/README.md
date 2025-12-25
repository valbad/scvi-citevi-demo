# Data

This directory contains (or references) the datasets used in this project to
illustrate scVI-like and CITE-seq (RNA + protein) models.

No large datasets are tracked in the GitHub repository.
All data must be downloaded locally (for example on Google Drive when using
Google Colab).

---

## 1. scRNA-seq data (RNA only) — Cortex

We use the Cortex scRNA-seq dataset published by the Linnarsson lab, which is
commonly used as a benchmark dataset for scVI.

### Download instructions

From a Colab notebook (or any environment with `wget` available):

```python
import os

url = "https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt"
out_path = "data/cortex_rna.txt"

if not os.path.exists(out_path):
    !wget -O {out_path} {url}
```

The downloaded file contains a raw expression matrix that will be parsed and
converted into PyTorch tensors by the data-loading utilities defined in ```src/data.py```.

## 2. CITE-seq data — PBMC 2019 (RNA + protein)

We use a CITE-seq dataset of human peripheral blood mononuclear cells (PBMCs)
released in 2019.  
CITE-seq (Cellular Indexing of Transcriptomes and Epitopes by sequencing)
combines single-cell RNA sequencing with antibody-derived tag (ADT) protein
measurements, enabling joint analysis of transcriptomic and surface protein
expression.

This type of multimodal data is also used in the Kaggle competition  
*Open Problems – Multimodal Single-Cell Integration*:
https://www.kaggle.com/competitions/open-problems-multimodal/overview

### Dataset description and reference

The data originate from the following study:

Stuart T, Butler A, Hoffman P, Hafemeister C, *et al.*  
**Comprehensive Integration of Single-Cell Data**.  
*Cell*, 2019; 177(7):1888–1902.e21.  
PMID: 31178118  
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6687398/

The corresponding GEO accession is:

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128639

---

### Download via Kaggle

The dataset is downloaded using `kagglehub`.  
Note that KaggleHub stores datasets in a local cache directory; therefore,
the relevant files are explicitly copied into the project `data/` directory
for reproducibility and consistency.

```python
import kagglehub
import os
import shutil

# 1. Download to KaggleHub cache
cache_path = kagglehub.dataset_download(
    "alexandervc/citeseq-scrnaseq-proteins-human-pbmcs-2019"
)

print("Cached dataset path:", cache_path)

# 2. Copy relevant files into data/
target_dir = "data/pbmc_citeseq"
os.makedirs(target_dir, exist_ok=True)

for subdir in os.listdir(cache_path):
    src = os.path.join(cache_path, subdir)
    dst = os.path.join(target_dir, subdir)
    if os.path.isdir(src) and not os.path.exists(dst):
        shutil.copytree(src, dst)

print("Dataset copied to:", target_dir)
```

## Expected structure

After running this pipeline the dataset should be structured as follows: 

``` 
data/
├── README.md
├── pbmc_citeseq/
|   ├── GSM3681518_MNC_RNA_counts.tsv/
|   └── GSM3681519_MNC_ADT_counts.tsv/
└── cortex_rna.txt
```