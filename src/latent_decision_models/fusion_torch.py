"""Small PyTorch fusion network (optional ``[deep]`` extra; CPU).

Architecture: neural features -> linear layer (``eeg_units``), concatenated with the
behavior features -> ``hidden`` ReLU units with dropout -> 1 output. Trained full-batch
with AdamW on mean squared error; early stopping on ``n_val_subjects`` training groups
held out for validation (never the test data); predictions are averaged over
``n_seeds`` seeds. Inputs are standardised with training statistics.
"""

from __future__ import annotations

import numpy as np


def _net(n_e: int, n_b: int, cfg: dict):
    import torch
    from torch import nn

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.eeg = nn.Linear(n_e, cfg["eeg_units"])
            self.head = nn.Sequential(nn.Linear(cfg["eeg_units"] + n_b, cfg["hidden"]),
                                      nn.ReLU(), nn.Dropout(cfg["dropout"]),
                                      nn.Linear(cfg["hidden"], 1))

        def forward(self, b, e):
            return self.head(torch.cat([self.eeg(e), b], dim=1)).squeeze(1)

    return Fusion()


def fit_predict_mlp(B_tr, E_tr, y_tr, groups_tr, B_te, E_te, cfg: dict,
                    seed: int = 0) -> np.ndarray:
    try:
        import torch
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError("the MLP needs PyTorch: pip install "
                          "'latent-decision-models[deep]'") from e
    torch.set_num_threads(1)

    def std(a_tr, a_te):
        mu, sd = a_tr.mean(axis=0), a_tr.std(axis=0)
        sd[sd == 0] = 1.0
        return (a_tr - mu) / sd, (a_te - mu) / sd

    B_tr, B_te = std(np.asarray(B_tr, float), np.asarray(B_te, float))
    E_tr, E_te = std(np.asarray(E_tr, float), np.asarray(E_te, float))
    y_tr = np.asarray(y_tr, float)
    y_mu, y_sd = y_tr.mean(), y_tr.std() or 1.0
    groups = np.unique(groups_tr)
    rng = np.random.default_rng(seed)
    n_val = cfg.get("n_val_subjects", 2)
    if len(groups) <= n_val:
        raise ValueError("need more training groups than validation groups")
    val = np.isin(groups_tr, rng.choice(groups, n_val, replace=False))

    def t(a):
        return torch.as_tensor(a, dtype=torch.float32)

    tb, te_, ty = t(B_tr[~val]), t(E_tr[~val]), t((y_tr[~val] - y_mu) / y_sd)
    vb, ve, vy = t(B_tr[val]), t(E_tr[val]), t((y_tr[val] - y_mu) / y_sd)
    preds = []
    for k in range(cfg.get("n_seeds", 5)):
        torch.manual_seed(seed * 1000 + k)
        net = _net(E_tr.shape[1], B_tr.shape[1], cfg)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
        best, best_state, wait = np.inf, None, 0
        for _ in range(cfg["max_epochs"]):
            net.train()
            opt.zero_grad()
            loss = torch.mean((net(tb, te_) - ty) ** 2)
            loss.backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                v = float(torch.mean((net(vb, ve) - vy) ** 2))
            if v < best - 1e-6:
                best, wait = v, 0
                best_state = {n: p.detach().clone() for n, p in net.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            preds.append(net(t(B_te), t(E_te)).numpy() * y_sd + y_mu)
    return np.mean(preds, axis=0)
