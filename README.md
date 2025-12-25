# scVI / CITE-seq — Variational Autoencoders for Single-Cell Data

This repository contains a minimal, self-contained implementation of
variational autoencoder models for single-cell data, inspired by **scVI**
and its multimodal extension for **CITE-seq** (RNA + protein).

The code is written in PyTorch and follows a clear separation between
data handling, model definition, probabilistic modeling, training, and
evaluation.

---

## Context

This project was developed as part of the course:

**Introduction to Probabilistic Graphical Models**  
Pierre Latouche — Pierre-Alexandre Mattei  
MVA (Master Mathématiques, Vision, Apprentissage)

Course webpage:  
https://lmbp.uca.fr/~latouche/mva/IntroductiontoProbabilisticGraphicalModelsMVA.html

The implementation aims to connect:
- probabilistic graphical modeling concepts (latent variables, ELBO),
- variational inference,
- deep generative models applied to real biological data.

---

## Models implemented

- **SCVI**  
  RNA-only variational autoencoder with:
  - latent biological variable `z`
  - latent library size variable `l`
  - batch-specific priors
  - ZINB likelihood for gene expression

- **CiteVI**  
  Multimodal extension for CITE-seq data:
  - shared latent variable `z`
  - RNA modeled with ZINB
  - proteins (ADT) modeled with a negative binomial distribution

Both models are trained by maximizing an evidence lower bound (ELBO).

---

## Repository structure

```text
src/
├── data.py              # dataset definition and data loaders
├── building_blocks.py   # neural network components (encoders / decoders)
├── distributions.py     # KL terms, likelihoods, ELBOs
├── models.py            # SCVI and CiteVI models
├── training.py          # training loops
└── evaluation.py        # diagnostics and visualization

notebooks/
├── scVI.ipynb
└── citeVI.ipynb

data/
├── README.md            # data download instructions
├── cortex_rna.txt       # RNA-only dataset (not versioned)
└── pbmc_citeseq/        # CITE-seq dataset (not versioned)
```

Raw datasets are not tracked in the repository.

---

## Data
- **Cortex scRNA-seq** (RNA-only): Used as a benchmark dataset for scVI-style models.
- **PBMC 2019 CITE-seq** (RNA + protein): Human PBMC dataset combining transcriptomic and surface protein measurements.

Detailed download instructions are provided in ```data/README.md```.

---

## Usage

Training and evaluation are demonstrated in the notebooks located in
the ```notebooks/``` directory.

Typical workflow:

1. Download the data (see ```data/README.md```)
2. Run the corresponding notebook
3. Train the model using the provided training loop
4. Evaluate the learned latent space via reconstruction metrics and t-SNE

---

## Scope and limitations

This repository is intended for:
- educational purposes,
- understanding variational inference in practice,
- illustrating probabilistic modeling choices for single-cell data.

It is not meant to be a production-ready alternative to ```scvi-tools```.

---

## References

- Lopez R. et al., *Deep generative modeling for single-cell transcriptomics*
- Stuart T. et al., *Comprehensive Integration of Single-Cell Data, Cell* (2019)
- Kingma & Welling, *Auto-Encoding Variational Bayes*
