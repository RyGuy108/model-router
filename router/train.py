"""Train + calibrate PassNet from eval-harness rows (sim/world.py output, or rows exported from the
service's Postgres feedback tables with the same shape).

    python -m router.train --data data/eval.jsonl --out models/passnet.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from router.catalog import MODEL_INDEX, QUALITY_BARS
from router.model import N_BARS, N_MODELS, PassNet, collate, save_bytes


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def split(rows: list[dict], seed: int = 0, frac=(0.33, 0.27)):
    """train / val / test. Train is kept at eval-harness scale (~30k requests); val and test are
    large so that 'equal quality' comparisons have tight confidence intervals."""
    rs = sorted(rows, key=lambda r: hashlib.md5(f"{seed}:{r['id']}".encode()).hexdigest())
    a, b = int(len(rs) * frac[0]), int(len(rs) * (frac[0] + frac[1]))
    return rs[:a], rs[a:b], rs[b:]


def targets(rows: list[dict]):
    """(N, M, B) mean pass label, (N, M, B) mask of observed (model, bar) labels.
    Eval-harness rows label every bar of every rendered model; production feedback rows label one
    bar of one model, so the mask is per (model, bar). Optional row["weight"] scales the mask."""
    y = np.zeros((len(rows), N_MODELS, N_BARS), dtype=np.float32)
    m = np.zeros((len(rows), N_MODELS, N_BARS), dtype=np.float32)
    for i, r in enumerate(rows):
        w = float(r.get("weight", 1.0))
        for name, outs in r["outcomes"].items():
            j = MODEL_INDEX[name]
            for k, bar in enumerate(QUALITY_BARS):
                obs = [o[bar] for o in outs if bar in o]
                if obs:
                    y[i, j, k] = np.mean(obs)
                    m[i, j, k] = w
    return torch.from_numpy(y), torch.from_numpy(m)


def masked_bce(z, y, m):
    return (F.binary_cross_entropy_with_logits(z, y, reduction="none") * m).sum() / m.sum().clamp_min(1e-9)


@torch.no_grad()
def predict_logits(net: PassNet, reqs: list[dict], calibrated: bool, bs: int = 2048) -> torch.Tensor:
    net.eval()
    out = []
    for i in range(0, len(reqs), bs):
        out.append(net.logits(*collate(reqs[i:i + bs]), calibrated=calibrated))
    return torch.cat(out)


def fit_temperature(net: PassNet, reqs, y, m):
    """Per-model temperature scaling on held-out data (Guo et al. 2017)."""
    z = predict_logits(net, reqs, calibrated=False)
    log_t = torch.zeros(N_MODELS, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = masked_bce(z / torch.exp(log_t)[None, :, None], y, m)
        loss.backward()
        return loss

    opt.step(closure)
    net.log_temp.copy_(log_t.detach())
    return torch.exp(log_t).detach()


def train(train_rows, val_rows, epochs=8, bs=256, lr=2e-3, seed=0, log=print, init: PassNet | None = None,
          arch: dict | None = None, wd: float = 0.0) -> PassNet:
    torch.manual_seed(seed)
    random.seed(seed)
    net = PassNet(**(arch or {}))
    if init is not None:  # warm start (fine-tuning on fresh feedback)
        net.load_state_dict(init.state_dict())
        net.bias.zero_()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    reqs = [r["req"] for r in train_rows]
    y, m = targets(train_rows)
    vreqs = [r["req"] for r in val_rows]
    vy, vm = targets(val_rows)
    best, best_state = 1e9, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(reqs))
        t0, tot = time.time(), 0.0
        for i in range(0, len(reqs), bs):
            idx = perm[i:i + bs]
            z = net.logits(*collate([reqs[j] for j in idx]), calibrated=False)
            loss = masked_bce(z, y[idx], m[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        vl = masked_bce(predict_logits(net, vreqs, calibrated=False), vy, vm).item()
        log(f"epoch {ep + 1}: train {tot / len(reqs):.4f}  val {vl:.4f}  ({time.time() - t0:.1f}s)")
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    temps = fit_temperature(net, vreqs, vy, vm)
    log("temperatures: " + " ".join(f"{t:.2f}" for t in temps.tolist()))
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval.jsonl")
    ap.add_argument("--out", default="models/passnet.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--buckets-log2", type=int, default=18)
    ap.add_argument("--wd", type=float, default=0.0)
    a = ap.parse_args()
    arch = {"emb_dim": a.emb, "hidden": a.hidden, "buckets": 1 << a.buckets_log2}
    torch.set_num_threads(4)
    rows = load_rows(a.data)
    tr, va, te = split(rows)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")
    net = train(tr, va, epochs=a.epochs, seed=a.seed, arch=arch, wd=a.wd)
    version = time.strftime("%Y%m%d-%H%M%S")
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(save_bytes(net, {"version": version, "data": a.data, "n_train": len(tr), "arch": arch}))
    print(f"saved {p} version {version}")


if __name__ == "__main__":
    main()
