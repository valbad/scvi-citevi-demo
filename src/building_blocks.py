# src/building_blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchEmbedding(nn.Module):
    def __init__(self, n_batches, emb_dim=10):
        super().__init__()
        self.emb = nn.Embedding(n_batches, emb_dim)

    def forward(self, batch):
        return self.emb(batch)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, n_layers=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class EncoderZ_RNA(nn.Module):
    def __init__(self, n_genes, n_batches, latent_dim,
                 hidden_dim=128, emb_dim=10):
        super().__init__()
        self.batch_emb = BatchEmbedding(n_batches, emb_dim)
        self.mlp = MLP(n_genes + emb_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, batch):
        h = torch.cat([x, self.batch_emb(batch)], dim=1)
        h = self.mlp(h)
        return self.mu(h), self.logvar(h)
    
  
class EncoderZ_Joint(nn.Module):
    def __init__(self, n_genes, n_proteins, n_batches, latent_dim,
                 hidden_dim=128, emb_dim=10):
        super().__init__()
        self.batch_emb = BatchEmbedding(n_batches, emb_dim)
        self.mlp = MLP(n_genes + n_proteins + emb_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y, batch):
        h = torch.cat([x, y, self.batch_emb(batch)], dim=1)
        h = self.mlp(h)
        return self.mu(h), self.logvar(h)

class EncoderL(nn.Module):
    def __init__(self, n_genes, n_batches,
                 hidden_dim=128, emb_dim=10):
        super().__init__()
        self.batch_emb = BatchEmbedding(n_batches, emb_dim)
        self.mlp = MLP(n_genes + emb_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, 1)
        self.logvar = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch):
        h = torch.cat([x, self.batch_emb(batch)], dim=1)
        h = self.mlp(h)
        return self.mu(h).squeeze(1), self.logvar(h).squeeze(1)

class DecoderRNA(nn.Module):
    def __init__(self, latent_dim, n_batches, n_genes,
                 hidden_dim=128, emb_dim=10):
        super().__init__()
        self.batch_emb = BatchEmbedding(n_batches, emb_dim)
        self.mlp = MLP(latent_dim + emb_dim, hidden_dim)
        self.rho_logits = nn.Linear(hidden_dim, n_genes)
        self.pi_logits = nn.Linear(hidden_dim, n_genes)

    def forward(self, z, batch):
        h = torch.cat([z, self.batch_emb(batch)], dim=1)
        h = self.mlp(h)
        rho = torch.softmax(self.rho_logits(h), dim=1)
        pi = torch.sigmoid(self.pi_logits(h))
        return rho, pi

class DecoderADT(nn.Module):
    def __init__(self, latent_dim, n_batches, n_proteins,
                 hidden_dim=128, emb_dim=10):
        super().__init__()
        self.batch_emb = BatchEmbedding(n_batches, emb_dim)
        self.mlp = MLP(latent_dim + emb_dim, hidden_dim)
        self.mu_raw = nn.Linear(hidden_dim, n_proteins)

    def forward(self, z, batch):
        h = torch.cat([z, self.batch_emb(batch)], dim=1)
        h = self.mlp(h)
        mu = F.softplus(self.mu_raw(h))
        return torch.clamp(mu, min=1e-4, max=1e4)
