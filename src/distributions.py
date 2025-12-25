# src/distributions.py 

import torch

def kl_normal(mu, logvar):
    return 0.5 * torch.sum(
        torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1
    )

def kl_l(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (
        logvar_p - logvar_q +
        (var_q + (mu_q - mu_p)**2) / var_p - 1.0
    )

def zinb_log_likelihood(x, mu, theta, pi, eps=1e-8):
    t1 = torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1)
    t2 = theta * (torch.log(theta + eps) - torch.log(mu + theta + eps))
    t3 = x * (torch.log(mu + eps) - torch.log(mu + theta + eps))
    nb_case = t1 + t2 + t3

    zero_mask = (x < eps)
    nb_zero = theta * (torch.log(theta + eps) - torch.log(mu + theta + eps))

    zinb_zero = torch.log(pi + (1 - pi) * torch.exp(nb_zero) + eps)
    zinb_nonzero = torch.log(1 - pi + eps) + nb_case

    return torch.where(zero_mask, zinb_zero, zinb_nonzero)

def nb_log_likelihood(x, mu, theta, eps=1e-8):
    mu = torch.clamp(mu, min=eps, max=1e6)
    theta = torch.clamp(theta, min=eps, max=1e6)

    t1 = torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1)
    log_theta_mu = torch.log(theta + mu + eps)
    t2 = theta * (torch.log(theta + eps) - log_theta_mu)
    t3 = x * (torch.log(mu + eps) - log_theta_mu)

    return t1 + t2 + t3

def elbo_scvi(model, batch, outputs):
    x = batch["x"]
    batch_idx = batch["batch"]

    recon_x = zinb_log_likelihood(
        x,
        outputs["mu_x"],
        torch.exp(outputs["log_theta_x"]),
        outputs["pi_x"]
    ).sum(dim=1)

    klz = kl_normal(outputs["mu_z"], outputs["logvar_z"])

    kll = kl_l(
        outputs["mu_l"],
        outputs["logvar_l"],
        model.mu_l_batch[batch_idx],
        model.logvar_l_batch[batch_idx]
    )

    elbo = recon_x - klz - kll
    loss = -elbo.mean()

    return loss, {
        "loss": loss.item(),
        "recon_x": -recon_x.mean().item(),
        "kl_z": klz.mean().item(),
        "kl_l": kll.mean().item(),
    }

def elbo_citevi(model, batch, outputs, weight_adt=0.1):
    x = batch["x"]
    y = batch["y"]
    batch_idx = batch["batch"]

    recon_x = zinb_log_likelihood(
        x,
        outputs["mu_x"],
        torch.exp(outputs["log_theta_x"]),
        outputs["pi_x"]
    ).sum(dim=1)

    recon_y = nb_log_likelihood(
        y,
        outputs["mu_y"],
        torch.exp(outputs["log_theta_y"])
    ).sum(dim=1)

    klz = kl_normal(outputs["mu_z"], outputs["logvar_z"])

    kll = kl_l(
        outputs["mu_l"],
        outputs["logvar_l"],
        model.mu_l_batch[batch_idx],
        model.logvar_l_batch[batch_idx]
    )

    elbo = recon_x + weight_adt * recon_y - klz - kll
    loss = -elbo.mean()

    return loss, {
        "loss": loss.item(),
        "recon_x": -recon_x.mean().item(),
        "recon_y": -recon_y.mean().item(),
        "kl_z": klz.mean().item(),
        "kl_l": kll.mean().item(),
    }
