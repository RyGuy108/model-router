"""Retrain from production feedback, gate, and hot-swap.

    python -m router.retrain [--eval data/eval.jsonl] [--min-feedback 2000] [--dry-run]

1. Export (decision.request, routed model, passed at the requested bar) from Postgres.
   Each row labels one (model, bar) cell; the per-cell mask in router/train.py handles that.
   Rows are importance-weighted by 1/propensity (clipped) so exploration traffic, which the
   greedy policy would rarely produce, counts for what it represents.
2. Fine-tune the active model on feedback (oldest 70%; next 15% for early stopping and temperature) mixed with a slice of the original
   eval set, so models that get little production traffic don't drift on nothing.
3. Gate: the candidate must beat the active model *including its online bias correction* on
   the newest 15% of feedback (never used for fitting) (log loss). Only then register + activate (Redis pubsub hot swap).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import psycopg
import redis
import torch

from router.catalog import MODEL_INDEX
from router.model import load_bytes, save_bytes
from router.registry import NS, PG, RD, activate
from router.train import load_rows, predict_logits, split, train

BAR_IX = {"draft": 0, "standard": 1, "premium": 2}


def export_feedback(conn) -> list[dict]:
    q = ("SELECT d.id, d.request, f.model, f.passed, CASE WHEN f.shadow THEN d.shadow_propensity ELSE d.propensity END "
         "FROM feedback f JOIN decisions d ON d.id = f.decision_id ORDER BY f.received_at")
    rows = []
    for did, req, model, passed, prop in conn.execute(q):
        prop = prop or 1.0
        rows.append({"id": f"fb-{did}-{model}", "req": req, "weight": float(min(20.0, 1.0 / max(prop, 1e-6))),
                     "outcomes": {model: [{req["quality"]: int(passed)}]}})
    return rows


def logloss(net, rows, bias: np.ndarray | None = None) -> float:
    z = predict_logits(net, [r["req"] for r in rows], calibrated=True).numpy()
    tot = 0.0
    for i, r in enumerate(rows):
        (model, outs), = r["outcomes"].items()
        bar = r["req"]["quality"]
        j = MODEL_INDEX[model]
        zz = z[i, j, BAR_IX[bar]] + (bias[j] if bias is not None else 0.0)
        p = float(np.clip(1 / (1 + np.exp(-zz)), 1e-6, 1 - 1e-6))
        y = outs[0][bar]
        tot += -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return tot / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval.jsonl")
    ap.add_argument("--eval-sample", type=int, default=8000)
    ap.add_argument("--min-feedback", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("ROUTER_TRAIN_THREADS", "4")))
    t0 = time.time()
    with psycopg.connect(PG) as c:
        fb = export_feedback(c)
        ver, blob, tau = c.execute("SELECT version, artifact, tau FROM model_registry WHERE active").fetchone()
    if len(fb) < a.min_feedback:
        print(json.dumps({"status": "skipped", "feedback": len(fb)}))
        return
    cur, meta = load_bytes(bytes(blob))
    r = redis.from_url(RD)
    bias = np.zeros(len(MODEL_INDEX))
    for k, v in r.hgetall(f"{NS}:bias").items():
        if k.decode() in MODEL_INDEX:
            bias[MODEL_INDEX[k.decode()]] = float(v)

    c1, c2 = int(len(fb) * 0.7), int(len(fb) * 0.85)
    fb_train, fb_cal, fb_hold = fb[:c1], fb[c1:c2], fb[c2:]  # cal: early stop + temperature; hold: gate only
    ev_tr, ev_va, _ = split(load_rows(a.eval))
    rng = random.Random(0)
    base = rng.sample(ev_tr, min(a.eval_sample, len(ev_tr)))
    cand = train(base + fb_train * 2, rng.sample(ev_va, 1500) + fb_cal, epochs=a.epochs, lr=1e-3, init=cur, arch=meta.get("arch"),
                 log=lambda s: print("  " + s))
    old_ll = logloss(cur, fb_hold, bias)
    new_ll = logloss(cand, fb_hold)
    version = time.strftime("%Y%m%d-%H%M%S") + "-ft"
    report = {"status": "rejected", "feedback": len(fb), "active": ver, "holdout_logloss_active_with_bias": old_ll,
              "holdout_logloss_candidate": new_ll, "seconds": round(time.time() - t0, 1)}
    if new_ll < old_ll and not a.dry_run:
        b2 = save_bytes(cand, {"version": version, "parent": ver, "n_feedback": len(fb), "arch": meta.get("arch", {})})
        with psycopg.connect(PG) as c:
            c.execute("INSERT INTO model_registry (version, artifact, tau, metrics) VALUES (%s,%s,%s,%s)",
                      (version, b2, json.dumps(tau), json.dumps({"retrain": report})))
        activate(version)
        report.update(status="activated", version=version)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
