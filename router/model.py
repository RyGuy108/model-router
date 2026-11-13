"""PyTorch pass-probability model.

One network scores every model in the catalog at once:
  text  : hashed uni/bigrams -> EmbeddingBag (mean) -> 64-d
  dense : task / resolution / aspect / seconds / quoted-text features -> 32-d
  trunk : 2-layer MLP -> 128-d shared request representation
  head  : per catalog model, a scalar latent quality a_m(x) and an ordinal (cumulative-link)
          output P(pass at bar b) = sigmoid((a_m(x) - c_{m,b}) / T_m), c_m increasing in b.
The ordinal head guarantees P(premium) <= P(standard) <= P(draft) for every model, and T_m is a
per-model temperature fitted after training on held-out data (calibration).
"""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from router.catalog import ALL_MODELS, QUALITY_BARS
from router.features import DENSE_DIM, HASH_BUCKETS, dense, tokens

N_MODELS = len(ALL_MODELS)
N_BARS = len(QUALITY_BARS)


class PassNet(nn.Module):
    def __init__(self, emb_dim: int = 64, hidden: int = 128, buckets: int = HASH_BUCKETS):
        super().__init__()
        self.emb = nn.EmbeddingBag(buckets, emb_dim, mode="mean", padding_idx=0)
        self.dense = nn.Sequential(nn.Linear(DENSE_DIM, 32), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(emb_dim + 32, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.latent = nn.Linear(hidden, N_MODELS)
        self.cut0 = nn.Parameter(torch.zeros(N_MODELS))
        self.cut_gaps = nn.Parameter(torch.zeros(N_MODELS, N_BARS - 1))
        self.register_buffer("log_temp", torch.zeros(N_MODELS))
        self.register_buffer("bias", torch.zeros(N_MODELS))  # online correction, applied by the service from Redis, see router/service.py

    def cuts(self) -> torch.Tensor:  # (M, B), increasing along B
        return torch.cat([self.cut0[:, None], self.cut0[:, None] + torch.cumsum(F.softplus(self.cut_gaps), 1)], 1)

    def logits(self, tok: torch.Tensor, offsets: torch.Tensor, dn: torch.Tensor, calibrated: bool = True):
        tok = (tok - 1) % (self.emb.num_embeddings - 1) + 1  # fold global hash space into this table
        h = self.trunk(torch.cat([self.emb(tok, offsets), self.dense(dn)], 1))
        a = self.latent(h)  # (N, M)
        z = a[:, :, None] - self.cuts()[None]  # (N, M, B)
        if calibrated:
            z = z / torch.exp(self.log_temp)[None, :, None] + self.bias[None, :, None]
        return z

    def forward(self, tok, offsets, dn):
        return torch.sigmoid(self.logits(tok, offsets, dn))


def collate(reqs: list[dict]):
    toks, offs, o = [], [], 0
    for r in reqs:
        t = tokens(r["prompt"])
        offs.append(o)
        toks.extend(t)
        o += len(t)
    return (torch.tensor(toks, dtype=torch.long), torch.tensor(offs, dtype=torch.long),
            torch.from_numpy(np.stack([dense(r) for r in reqs])))


def save_bytes(net: PassNet, meta: dict) -> bytes:
    buf = io.BytesIO()
    torch.save({"state": net.state_dict(), "meta": meta}, buf)
    return buf.getvalue()


def load_bytes(b: bytes) -> tuple[PassNet, dict]:
    ck = torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    net = PassNet(**ck["meta"].get("arch", {}))
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["meta"]
