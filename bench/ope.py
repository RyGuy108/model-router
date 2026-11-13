"""Off-policy evaluation: score a new routing policy from production logs, without deploying it.

    # 1. serve with uniform exploration so every model has a logged propensity > 0
    ROUTER_EPSILON=0.1 ROUTER_EXPLORE=uniform uvicorn router.service:app --workers 3
    # 2. send traffic + feedback through it (simulated renders), keeping the sim seeds as ground truth
    python -m bench.ope collect --n 20000
    # 3. evaluate candidate policies from the Postgres decision log
    python -m bench.ope evaluate

Each candidate policy is a deterministic function of what the router logged for a request (every
eligible model's P(pass) and credit price), so its credits are known exactly; only its pass rate
needs estimating. Estimators:
  IPS   mean(1[a = pi(x)] * y / propensity)
  SNIPS self-normalised IPS (lower variance, slight bias)
  DM    mean(P_model(pass | x, pi(x)))  -- trusts the router's own probabilities
  DR    DM + IPS-weighted correction of the model's residual on the logged action
Ground truth: the simulator renders pi(x) for every logged request with the same random seeds, i.e.
what an A/B test of pi on exactly this traffic would have measured.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib

import httpx
import numpy as np
import psycopg

from bench.drift import make_stream
from router.catalog import BY_NAME
from router.policy import premium_default
from router.registry import PG
from sim.world import judge, quality

SIDE = pathlib.Path("data/ope_truth.jsonl")
TENANT = "ope"


async def collect(url: str, n: int, conc: int, seed: int):
    stream = make_stream(n, seed)
    side = open(SIDE, "w")
    lock = asyncio.Lock()
    nxt = 0

    async def worker():
        nonlocal nxt
        async with httpx.AsyncClient(base_url=url, timeout=30) as c:
            while True:
                async with lock:
                    if nxt >= n:
                        return
                    i = nxt
                    nxt += 1
                req, demand, diff, noise, s = stream[i]
                body = {"task": req.task, "prompt": req.prompt, "resolution": req.resolution, "aspect_ratio": req.aspect_ratio,
                        "seconds": req.seconds, "has_image_input": req.has_image_input, "quality": req.quality, "tenant": TENANT}
                d = (await c.post("/v1/route", json=body)).json()
                ok = judge(quality(d["model"], demand, diff, noise), np.random.default_rng(s))[req.quality]
                await c.post("/v1/feedback", json={"decision_id": d["decision_id"], "passed": bool(ok)})
                side.write(json.dumps({"id": d["decision_id"], "demand": list(map(float, demand)), "diff": diff,
                                       "noise": noise, "seed": s}) + "\n")

    await asyncio.gather(*(worker() for _ in range(conc)))
    side.close()


def load_logs():
    truth = {}
    for line in SIDE.read_text().splitlines():
        t = json.loads(line)
        truth[t["id"]] = t
    with psycopg.connect(PG) as c:
        rows = c.execute("SELECT d.id::text, d.request, d.chosen, d.propensity, d.scores, f.passed FROM decisions d "
                         "JOIN feedback f ON f.decision_id = d.id AND NOT f.shadow WHERE d.tenant = %s", (TENANT,)).fetchall()
    return [r for r in rows if r[0] in truth], truth


# candidate policies: (logged scores, request) -> model
def lagrangian(mult):
    def pol(sc, req):
        el = [BY_NAME[m] for m in sc]
        prem = premium_default(req["task"], el, req["resolution"], req["seconds"])
        lam = mult * KAPPA[req["task"]][req["quality"]] * sc[prem.name]["c"]
        return min(sc, key=lambda m: (sc[m]["c"] - lam * sc[m]["p"], sc[m]["c"]))
    return pol


def threshold(tau):
    def pol(sc, req):
        ok = [m for m in sc if sc[m]["p"] >= tau]
        return min(ok, key=lambda m: sc[m]["c"]) if ok else max(sc, key=lambda m: sc[m]["p"])
    return pol


def premium(sc, req):
    return premium_default(req["task"], [BY_NAME[m] for m in sc], req["resolution"], req["seconds"]).name


def cheapest(sc, req):
    return min(sc, key=lambda m: sc[m]["c"])


KAPPA: dict = {}


def evaluate(n_boot: int = 300, seed: int = 0):
    global KAPPA
    KAPPA = json.loads(pathlib.Path("models/tau.json").read_text())["kappa"]
    rows, truth = load_logs()
    pols = {"premium": premium, "cheapest": cheapest, "threshold@0.95": threshold(0.95), "threshold@0.99": threshold(0.99)}
    for mult in (0.25, 0.5, 1, 2, 4):
        pols[f"lagrangian x{mult}"] = lagrangian(mult)
    n = len(rows)
    y = np.array([float(r[5]) for r in rows])
    prop = np.array([r[3] for r in rows])
    p_logged = np.array([r[4][r[2]]["p"] for r in rows])
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, n, n) for _ in range(n_boot)]
    report = {"n_logged": n, "logging_explore_rate": float(np.mean(prop < 0.5)), "policies": {}}
    for name, pol in pols.items():
        act = [pol(r[4], r[1]) for r in rows]
        match = np.array([a == r[2] for a, r in zip(act, rows)], dtype=float)
        p_act = np.array([r[4][a]["p"] for a, r in zip(act, rows)])
        credits = float(np.mean([r[4][a]["c"] for a, r in zip(act, rows)]))
        w = match / prop
        ips_terms = w * y
        dr_terms = p_act + w * (y - p_logged)
        true = np.array([judge(quality(a, np.array(truth[r[0]]["demand"]), truth[r[0]]["diff"], truth[r[0]]["noise"]),
                               np.random.default_rng(truth[r[0]]["seed"]))[r[1]["quality"]] for a, r in zip(act, rows)], float)

        def est(idx):
            wi = w[idx]
            return {"IPS": float(np.mean(ips_terms[idx])), "SNIPS": float(np.sum(wi * y[idx]) / max(np.sum(wi), 1e-9)),
                    "DM": float(np.mean(p_act[idx])), "DR": float(np.mean(dr_terms[idx]))}
        point = est(np.arange(n))
        bs = [est(b) for b in boots]
        out = {"credits_per_request": credits, "true_pass": float(true.mean()), "match_rate": float(match.mean()),
               "estimates": {}}
        for k in point:
            lo, hi = np.percentile([b[k] for b in bs], [2.5, 97.5])
            out["estimates"][k] = {"value": point[k], "ci95": [float(lo), float(hi)], "error": point[k] - float(true.mean()),
                                   "covers_truth": bool(lo <= true.mean() <= hi)}
        report["policies"][name] = out
        e = out["estimates"]
        print(f"{name:18s} credits {credits:7.1f}  true {true.mean():.4f}  " + "  ".join(
            f"{k} {e[k]['value']:.4f}{'' if e[k]['covers_truth'] else '*'}" for k in ("IPS", "SNIPS", "DM", "DR")))
    others = [v for k, v in report["policies"].items() if k != "lagrangian x1"]  # x1 is the logging policy's greedy part
    report["summary_excluding_logging_policy"] = {
        k: {"mean_abs_error": float(np.mean([abs(v["estimates"][k]["error"]) for v in others])),
            "ci_coverage": float(np.mean([v["estimates"][k]["covers_truth"] for v in others]))}
        for k in ("IPS", "SNIPS", "DM", "DR")}
    report["logged_pass_rate"] = float(y.mean())
    print(json.dumps(report["summary_excluding_logging_policy"]), "logged pass", round(float(y.mean()), 4))
    pathlib.Path("results/ope.json").write_text(json.dumps(report, indent=1))
    print("(* = 95% interval misses the true value)  wrote results/ope.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--url", default="http://localhost:8000")
    c.add_argument("--n", type=int, default=20000)
    c.add_argument("--conc", type=int, default=32)
    c.add_argument("--seed", type=int, default=123)
    sub.add_parser("evaluate")
    a = ap.parse_args()
    if a.cmd == "collect":
        asyncio.run(collect(a.url, a.n, a.conc, a.seed))
    else:
        evaluate()


if __name__ == "__main__":
    main()
