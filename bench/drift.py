"""Drift replay through the live service: does the router notice when models change after training?

    python -m bench.drift --url http://localhost:8000 --arm static|bias|bias_explore_retrain --out results/drift.jsonl

A fresh stream of simulated requests goes through POST /v1/route; the simulator renders on whatever
model the service picked (with the drifted hidden profiles) and the outcome goes back through
POST /v1/feedback, with credits_charged. After `--drift-at` requests two things change silently:
  * gpt-image-2 regresses (a bad provider-side update: base quality -0.25). Before the change the
    router sends it ~15% of all traffic. It is never the premium default, so the baseline is unaffected.
  * flux-schnell (5 credits) ships a better version (base +0.08, text +0.5, anatomy +0.3). The router
    rarely picks it, so it can only find out by exploring.
  * kling-3.0 actually bills 0.6x the catalog price
The premium default is unaffected, so 'equal quality' is still measured against it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
import time

import httpx
import numpy as np

from router.catalog import BY_NAME, PREMIUM_DEFAULT, Request
from router.policy import premium_default
from sim.world import ModelDrift, judge, quality, sample_request

DRIFT = {"gpt-image-2": ModelDrift("gpt-image-2", base_delta=-0.25),
         "flux-schnell": ModelDrift("flux-schnell", base_delta=0.08, skill_delta={"text": 0.5, "anatomy": 0.3})}
TRUE_COST = {"kling-3.0": 0.6}


def make_stream(n: int, seed: int):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        req, demand, _ = sample_request(rng)
        if not req.eligible():
            continue
        out.append((req, demand, float(nrng.normal(0, 0.12)), float(nrng.normal(0, 0.09)), int(nrng.integers(1 << 31))))
    return out


async def run(url, arm, n, drift_at, retrain_at, conc, seed, window):
    stream = make_stream(n, seed)
    results = [None] * n
    lock = asyncio.Lock()
    nxt = 0

    async def worker(end):
        nonlocal nxt
        async with httpx.AsyncClient(base_url=url, timeout=30) as c:
            while True:
                async with lock:
                    if nxt >= end:
                        return
                    i = nxt
                    nxt += 1
                req, demand, diff, noise, s = stream[i]
                body = {"task": req.task, "prompt": req.prompt, "resolution": req.resolution, "aspect_ratio": req.aspect_ratio,
                        "seconds": req.seconds, "has_image_input": req.has_image_input, "quality": req.quality, "tenant": arm}
                d = (await c.post("/v1/route", json=body)).json()
                drift = DRIFT if i >= drift_at else None
                rng = np.random.default_rng(s)
                m = d["model"]
                ok = judge(quality(m, demand, diff, noise, drift), rng)[req.quality]
                spec = BY_NAME[m]
                charged = spec.credits(req.resolution, req.seconds) * TRUE_COST.get(m, 1.0)
                await c.post("/v1/feedback", json={"decision_id": d["decision_id"], "passed": bool(ok), "credits_charged": charged})
                prem = premium_default(req.task, req.eligible(), req.resolution, req.seconds)
                pok = judge(quality(prem.name, demand, diff, noise, drift), np.random.default_rng(s))[req.quality]
                results[i] = {"i": i, "model": m, "ok": ok, "credits": charged, "explored": d["explored"],
                              "version": d["model_version"], "prem_ok": pok,
                              "prem_credits": prem.credits(req.resolution, req.seconds),
                              "regressed": m == "gpt-image-2", "upgraded": m == "flux-schnell", "repriced": m == "kling-3.0"}

    t0 = time.time()
    retrain_out = None
    for end in ([retrain_at, n] if retrain_at else [n]):
        await asyncio.gather(*(worker(end) for _ in range(conc)))
        if end == retrain_at:  # nightly job: traffic keeps the old model until the new one is activated
            p = await asyncio.create_subprocess_exec(sys.executable, "-m", "router.retrain", "--min-feedback", "1000",
                                                     stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await p.communicate()
            retrain_out = out.decode().strip().splitlines()[-1]
            await asyncio.sleep(1.5)  # let workers pick up the pubsub swap + bias reset
    wins = []
    for w in range(0, n, window):
        chunk = results[w:w + window]
        wins.append({"start": w, "pass": float(np.mean([r["ok"] for r in chunk])),
                     "prem_pass": float(np.mean([r["prem_ok"] for r in chunk])),
                     "credits_per_req": float(np.mean([r["credits"] for r in chunk])),
                     "prem_credits_per_req": float(np.mean([r["prem_credits"] for r in chunk])),
                     "regressed_share": float(np.mean([r["regressed"] for r in chunk])),
                     "upgraded_share": float(np.mean([r["upgraded"] for r in chunk])),
                     "repriced_share": float(np.mean([r["repriced"] for r in chunk])),
                     "explored": float(np.mean([r["explored"] for r in chunk])),
                     "versions": sorted({r["version"] for r in chunk})})
    post = results[drift_at:]
    summary = {"arm": arm, "n": n, "drift_at": drift_at, "seconds": round(time.time() - t0, 1),
               "post_drift_pass": float(np.mean([r["ok"] for r in post])),
               "post_drift_prem_pass": float(np.mean([r["prem_ok"] for r in post])),
               "post_drift_credits_saved": 1 - sum(r["credits"] for r in post) / sum(r["prem_credits"] for r in post),
               "last_window": wins[-1], "retrain": retrain_out, "windows": wins}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--drift-at", type=int, default=4000)
    ap.add_argument("--retrain-at", type=int, default=0)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--out")
    a = ap.parse_args()
    s = asyncio.run(run(a.url, a.arm, a.n, a.drift_at, a.retrain_at, a.conc, a.seed, a.window))
    print(json.dumps({k: v for k, v in s.items() if k != "windows"}, indent=1))
    for w in s["windows"]:
        print(w)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
