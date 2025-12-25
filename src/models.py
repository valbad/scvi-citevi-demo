# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .building_blocks import (
    EncoderZ_RNA, EncoderZ_Joint, EncoderL,
    DecoderRNA, DecoderADT
)

from .distributions import (
    kl_normal, kl_l,
    zinb_log_likelihood, nb_log_likelihood
)

class SCVI(nn.Module):
    def __init__(self, n_genes, n_batches,
                 mu_l_batch, var_l_batch, latent_dim=10):
        super().__init__()

        self.encoder_z = EncoderZ_RNA(n_genes, n_batches, latent_dim)
        self.encoder_l = EncoderL(n_genes, n_batches)
        self.decoder_rna = DecoderRNA(latent_dim, n_batches, n_genes)

        self.log_theta_rna = nn.Parameter(torch.zeros(n_genes))

        self.register_buffer("mu_l_batch", mu_l_batch)
        self.register_buffer(
            "logvar_l_batch", torch.log(var_l_batch + 1e-8)
        )
        self.register_buffer(
            "std_l_batch", torch.sqrt(var_l_batch + 1e-8)
        )

    def forward(self, batch_dict):
        x = batch_dict["x"]
        batch = batch_dict["batch"]

        mu_z, logvar_z = self.encoder_z(x, batch)
        z = mu_z + torch.exp(0.5 * logvar_z) * torch.randn_like(mu_z)

        mu_l, logvar_l = self.encoder_l(x, batch)
        log_l = mu_l + torch.exp(0.5 * logvar_l) * torch.randn_like(mu_l)

        low = self.mu_l_batch[batch] - 3 * self.std_l_batch[batch]
        high = self.mu_l_batch[batch] + 3 * self.std_l_batch[batch]
        log_l = torch.clamp(log_l, low, high)
        l = torch.exp(log_l)

        rho, pi = self.decoder_rna(z, batch)
        mu_x = l.unsqueeze(1) * rho

        return {
            "mu_x": mu_x,
            "pi_x": pi,
            "log_theta_x": self.log_theta_rna,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "mu_l": mu_l,
            "logvar_l": logvar_l,
        }

class CiteVI(nn.Module):
    def __init__(self, n_genes, n_proteins, n_batches,
                 mu_l_batch, var_l_batch, latent_dim=10):
        super().__init__()

        self.encoder_z = EncoderZ_Joint(
            n_genes, n_proteins, n_batches, latent_dim
        )
        self.encoder_l = EncoderL(n_genes, n_batches)

        self.decoder_rna = DecoderRNA(latent_dim, n_batches, n_genes)
        self.decoder_adt = DecoderADT(latent_dim, n_batches, n_proteins)

        self.log_theta_rna = nn.Parameter(torch.zeros(n_genes))
        self.log_theta_adt = nn.Parameter(torch.zeros(n_proteins))

        self.register_buffer("mu_l_batch", mu_l_batch)
        self.register_buffer(
            "logvar_l_batch", torch.log(var_l_batch + 1e-8)
        )
        self.register_buffer(
            "std_l_batch", torch.sqrt(var_l_batch + 1e-8)
        )

    def forward(self, batch_dict):
        x = batch_dict["x"]
        y = torch.log1p(batch_dict["y"])
        batch = batch_dict["batch"]

        mu_z, logvar_z = self.encoder_z(x, y, batch)
        z = mu_z + torch.exp(0.5 * logvar_z) * torch.randn_like(mu_z)

        mu_l, logvar_l = self.encoder_l(x, batch)
        log_l = mu_l + torch.exp(0.5 * logvar_l) * torch.randn_like(mu_l)

        low = self.mu_l_batch[batch] - 3 * self.std_l_batch[batch]
        high = self.mu_l_batch[batch] + 3 * self.std_l_batch[batch]
        log_l = torch.clamp(log_l, low, high)
        l = torch.exp(log_l)

        rho, pi = self.decoder_rna(z, batch)
        mu_x = l.unsqueeze(1) * rho
        mu_y = self.decoder_adt(z, batch)

        return {
            "mu_x": mu_x,
            "pi_x": pi,
            "log_theta_x": self.log_theta_rna,
            "mu_y": mu_y,
            "log_theta_y": self.log_theta_adt,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "mu_l": mu_l,
            "logvar_l": logvar_l,
        }

