# src/data.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SingleCellDataset(Dataset):
    """
    Dataset for scVI / CiteVI

    Stores:
      - RNA counts X (cells × genes)
      - optional ADT counts Y (cells × proteins)
      - batch index per cell
      - prior parameters for p(log l | batch)
    """

    def __init__(self, X, batch_indices, Y=None):
        super().__init__()

        self.X = X.float()                  # (N_cells, N_genes)
        self.batch = batch_indices.long()   # (N_cells,)
        self.Y = Y.float() if Y is not None else None

        # library size (RNA only)
        lib = torch.clamp(self.X.sum(dim=1), min=1.0)
        self.log_library = torch.log(lib)

        # batch-wise prior for log l
        self.n_batches = int(self.batch.max().item()) + 1
        mu = torch.zeros(self.n_batches)
        var = torch.zeros(self.n_batches)

        min_batch_val = self.batch.min().item()
        max_batch_val = self.batch.max().item()
        for b in range(min_batch_val, max_batch_val + 1):
            vals = self.log_library[self.batch == b]
            mu[b] = vals.mean()
            var[b] = vals.var(unbiased=False)

        self.mu_l_batch = mu
        self.var_l_batch = var
        self.logvar_l_batch = torch.log(var + 1e-8)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        out = {
            "x": self.X[idx],
            "batch": self.batch[idx]
        }
        if self.Y is not None:
            out["y"] = self.Y[idx]
        return out

def load_cortex_rna_dataset(
    path: str,
    device: torch.device = torch.device("cpu")
):
    """
    Load Cortex scRNA-seq dataset (RNA only).

    Returns
    -------
    dataset : SingleCellDataset
    labels : np.ndarray (cell-type labels)
    gene_names : np.ndarray
    """
    X = pd.read_csv(path, sep="\t", low_memory=False).T

    clusters = np.array(X[7], dtype=str)[2:]
    precise_clusters = np.array(X[0], dtype=str)[2:]

    celltypes, labels = np.unique(clusters, return_inverse=True)
    _, precise_labels = np.unique(precise_clusters, return_inverse=True)

    gene_names = np.array(X.iloc[0], dtype=str)[10:]

    X = X.loc[:, 10:]
    X = X.drop(X.index[0])

    expression = np.array(X, dtype=np.int32)[1:]

    # tensors
    X_tensor = torch.tensor(expression, dtype=torch.float32, device=device)
    batch = torch.zeros(X_tensor.shape[0], dtype=torch.long, device=device)

    dataset = SingleCellDataset(
        X=X_tensor,
        batch_indices=batch,
        Y=None
    )

    return dataset, gene_names, celltypes, labels, precise_labels

def load_pbmc_citeseq_dataset(
    rna_path: str,
    adt_path: str,
    device: torch.device = torch.device("cpu")
):
    """
    Load PBMC 2019 CITE-seq dataset.

    Parameters
    ----------
    rna_path : str
        Path to RNA counts TSV
    adt_path : str
        Path to ADT counts TSV

    Returns
    -------
    dataset : SingleCellDataset
    gene_names : np.ndarray
    protein_names : np.ndarray
    """
    import scanpy as sc
    ann_rna = sc.read_csv(rna_path, delimiter="\t")
    ann_adt = sc.read_csv(adt_path, delimiter="\t")

    X = ann_rna.X
    Y = ann_adt.X

    # convert to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()
    if hasattr(Y, "toarray"):
        Y = Y.toarray()
    X = X.T
    Y = Y.T

    # sanity check
    assert X.shape[0] == Y.shape[0], "RNA / ADT cell count mismatch"

    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    batch = torch.zeros(X.shape[0], dtype=torch.long, device=device)

    dataset = SingleCellDataset(
        X=X,
        Y=Y,
        batch_indices=batch
    )

    return dataset, ann_rna.var_names.to_numpy(), ann_adt.obs.index

