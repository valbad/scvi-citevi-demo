# src/utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def count_n_params(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model has", n_params, "trainable parameters")
    return 
    
def summary_plot(history):
    fig, ax = plt.subplots(1, len(history), figsize=(20, 5))

    ax[0].plot(history["recon_x"])
    ax[0].set_title("Reconstruction loss")

    ax[1].plot(history["loss"])
    ax[1].set_title("Total loss")

    ax[2].plot(history["kl_z"])
    ax[2].set_title("KL z")

    ax[3].plot(history["kl_l"])
    ax[3].set_title("KL l")

    if "recon_y" in history:
        ax[4].plot(history["recon_y"])
        ax[4].set_title("Reconstruction loss ADT")

    plt.show()

def predict_latents(model, dataloader, device="cuda"):
    model.eval()
    all_mu_z = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            all_mu_z.append(outputs["mu_z"].detach().cpu())

    return torch.cat(all_mu_z, dim=0).numpy()

def get_tsne_embedding(all_mu_z, perplexity=30):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="euclidean",
        random_state=42
    )
    return tsne.fit_transform(all_mu_z)

def tsne_cell_state_plot_with_labels(
    z_2d, labels, title="t-SNE of latent space"
):
    plt.figure(figsize=(10, 8))
    for l in np.unique(labels):
        mask = [i for i, lab in enumerate(labels) if lab == l]
        plt.scatter(
            z_2d[mask, 0],
            z_2d[mask, 1],
            label=l,
            alpha=0.5
        )
    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.show()

def tsne_cell_state_plot_with_scale(
    z_2d, scale, title="t-SNE of latent space"
):
    plt.figure(figsize=(10, 8))
    plt.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=scale,
        cmap="plasma"
    )
    plt.title(title)
    plt.axis("off")
    plt.colorbar()
    plt.show()
