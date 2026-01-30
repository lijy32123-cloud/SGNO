import argparse
import json
import os
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sgno import build_sgno_from_config

def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def _load_npz(path: str):
    d = np.load(path)
    keys = set(d.files)
    if "train" in keys and "test" in keys:
        return d["train"], d["test"]
    if "u" in keys:
        u = d["u"]
        n = u.shape[0]
        k = max(1, int(0.8 * n))
        return u[:k], u[k:]
    raise ValueError("NPZ must contain train and test or u")

def _ensure_layout(u: np.ndarray):
    if u.ndim < 4:
        raise ValueError("trajectory array must be at least 4D")
    return u

class TrajDataset(Dataset):
    def __init__(self, u: np.ndarray, initial_step: int):
        self.u = _ensure_layout(u).astype(np.float32, copy=False)
        self.initial_step = int(initial_step)
        self.n = self.u.shape[0]
        self.t = self.u.shape[1]
        if self.t <= self.initial_step:
            raise ValueError("T must be larger than initial_step")

    def __len__(self):
        return self.n * (self.t - self.initial_step)

    def __getitem__(self, idx: int):
        i = idx // (self.t - self.initial_step)
        j = idx % (self.t - self.initial_step)
        t0 = j
        t1 = j + self.initial_step
        x = self.u[i, t0:t1]
        y = self.u[i, t1]
        return torch.from_numpy(x), torch.from_numpy(y)

def _to_model_input(x: torch.Tensor, spatial_dim: int):
    if spatial_dim == 1:
        return x.permute(0, 1, 2, 3).contiguous()
    if spatial_dim == 2:
        return x.permute(0, 1, 2, 3, 4).contiguous()
    if spatial_dim == 3:
        return x.permute(0, 1, 2, 3, 4, 5).contiguous()
    raise ValueError("spatial_dim must be 1 2 or 3")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    spatial_dim = int(cfg["spatial_dim"])
    num_channels = int(cfg["num_channels"])
    num_points = int(cfg.get("num_points", 0))
    network_config = str(cfg["network_config"])
    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 4))
    epochs = int(train_cfg.get("epochs", 5))
    lr = float(train_cfg.get("lr", 1e-3))
    seed = int(train_cfg.get("seed", 0))

    _set_seed(seed)

    u_train, u_test = _load_npz(args.data)

    initial_step = 1
    for part in network_config.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip().lower() == "initial_step":
                initial_step = int(float(v.strip()))
                break

    ds = TrajDataset(u_train, initial_step=initial_step)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = build_sgno_from_config(network_config, spatial_dim, num_points, num_channels).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "best.pt")
    state_path = os.path.join(args.out, "state.json")

    best = math.inf
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y in dl:
            x = x.to(args.device)
            y = y.to(args.device)
            x_in = x
            y_hat = model(x_in)
            loss = loss_fn(y_hat, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        train_loss = total / max(1, n)

        model.eval()
        with torch.no_grad():
            u = torch.from_numpy(u_test[: min(4, u_test.shape[0])].astype(np.float32, copy=False))
            if u.ndim == 4:
                u0 = u[:, :initial_step]
            else:
                u0 = u[:, :initial_step]
            x0 = u0.to(args.device)
            y0 = torch.from_numpy(u_test[: min(4, u_test.shape[0]), initial_step].astype(np.float32, copy=False)).to(args.device)
            y_hat0 = model(x0)
            val_loss = float(loss_fn(y_hat0, y0).item())

        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "network_config": network_config, "spatial_dim": spatial_dim, "num_channels": num_channels, "num_points": num_points}, best_path)

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "best_val": best}, f)

    print(best_path)

if __name__ == "__main__":
    main()
