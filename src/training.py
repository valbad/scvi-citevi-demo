# src/training.py

import torch

def train_epoch(model, dataloader, optimizer, device, elbo_fn):
    model.train()
    logs = {}

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(batch)
        loss, stats = elbo_fn(model, batch, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in stats.items():
            logs[k] = logs.get(k, 0.0) + v

    for k in logs:
        logs[k] /= len(dataloader)

    return logs

def fit(
    model,
    dataloader,
    elbo_fn,
    epochs=50,
    lr=1e-3,
    device="cuda"
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {}
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        stats = train_epoch(
            model, dataloader, optimizer, device, elbo_fn
        )

        for k, v in stats.items():
            history.setdefault(k, []).append(v)

        msg = f"Epoch {epoch+1}/{epochs} | "
        msg += " | ".join(f"{k}: {v:.4f}" for k, v in stats.items())
        print(msg)

        if stats["loss"] < best_loss:
            best_loss = stats["loss"]
            best_state = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            }

    return history, best_state
