import argparse
import json
import math
import numpy as np
import torch
from sgno import build_sgno_from_config

def _load_npz(path: str):
    d = np.load(path)
    keys = set(d.files)
    if "test" in keys:
        return d["test"]
    if "u" in keys:
        return d["u"]
    raise ValueError("NPZ must contain test or u")

def _nrmse(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12):
    num = torch.linalg.norm(a - b)
    den = torch.linalg.norm(b) + eps
    return (num / den).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    spatial_dim = int(cfg["spatial_dim"])
    num_channels = int(cfg["num_channels"])
    num_points = int(cfg.get("num_points", 0))
    network_config = str(cfg["network_config"])

    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_sgno_from_config(network_config, spatial_dim, num_points, num_channels).to(args.device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    u = _load_npz(args.data).astype(np.float32, copy=False)
    u = torch.from_numpy(u)

    initial_step = 1
    for part in network_config.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip().lower() == "initial_step":
                initial_step = int(float(v.strip()))
                break

    n_traj = min(8, u.shape[0])
    t_total = u.shape[1]
    steps = min(args.steps, t_total - initial_step - 1)
    errs = []

    with torch.no_grad():
        for i in range(n_traj):
            hist = u[i, :initial_step].unsqueeze(0).to(args.device)
            pred = u[i, initial_step - 1].unsqueeze(0).to(args.device)
            for t in range(steps):
                y = model(hist)
                gt = u[i, initial_step + t].to(args.device)
                errs.append(_nrmse(y, gt))
                if initial_step == 1:
                    hist = y.unsqueeze(1)
                else:
                    hist = torch.cat([hist[:, 1:], y.unsqueeze(1)], dim=1)

    out = {
        "n_traj": int(n_traj),
        "steps": int(steps),
        "mean_nrmse": float(np.mean(errs) if errs else math.nan),
        "median_nrmse": float(np.median(errs) if errs else math.nan),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(args.out)

if __name__ == "__main__":
    main()
