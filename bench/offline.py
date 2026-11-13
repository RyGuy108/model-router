"""Offline benchmark: prediction quality and credits saved at equal quality.

    python -m bench.offline --model models/passnet.pt

Every policy is replayed on the held-out test split. Because the eval harness rendered every
request on every eligible model, each policy's choice has a real (simulated) judge outcome, so
pass rates are measured, not predicted. Thresholds are tuned on the validation split only.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from router.catalog import ALL_MODELS, BY_NAME, MODEL_INDEX, QUALITY_BARS, Request
from router.features import HASH_BUCKETS, dense, tokens
from router.model import load_bytes
from router.policy import choose, choose_cascade, choose_lagrangian, premium_default
from router.train import load_rows, predict_logits, split
from sim.world import true_pass_prob

BAR_IX = {b: i for i, b in enumerate(QUALITY_BARS)}


def req_obj(r):
    return Request(**r["req"])


# ---------------------------------------------------------------- predictors: rows -> (N, M, B) probs
def p_passnet(net, rows):
    return torch.sigmoid(predict_logits(net, [r["req"] for r in rows], calibrated=True)).numpy()


def p_oracle(rows):
    out = np.zeros((len(rows), len(ALL_MODELS), len(QUALITY_BARS)))
    for i, r in enumerate(rows):
        for name in r["outcomes"]:
            for b, bar in enumerate(QUALITY_BARS):
                out[i, MODEL_INDEX[name], b] = true_pass_prob(name, r["demand"], bar)
    return out


def _bucket(r):
    q = r["req"]
    return (q["task"], q["resolution"], '"' in q["prompt"], q.get("has_image_input", False),
            min(int(q.get("seconds") or 0), 15))


def make_bucket_prior(train_rows):
    """Non-ML baseline: per (model, bar, task, resolution, has-quoted-text, start-frame, duration)
    empirical pass rate from the eval set. This is what a careful engineer would build by hand."""
    s = collections.defaultdict(lambda: np.zeros(2))
    g = collections.defaultdict(lambda: np.zeros(2))
    for r in train_rows:
        k = _bucket(r)
        for name, outs in r["outcomes"].items():
            for bar in QUALITY_BARS:
                y = np.mean([o[bar] for o in outs])
                s[(name, bar) + k] += (y, 1)
                g[(name, bar, r["req"]["task"])] += (y, 1)

    def predict(rows):
        out = np.zeros((len(rows), len(ALL_MODELS), len(QUALITY_BARS)))
        for i, r in enumerate(rows):
            k = _bucket(r)
            for name in r["outcomes"]:
                for b, bar in enumerate(QUALITY_BARS):
                    a, n = s.get((name, bar) + k, (0, 0))
                    ga, gn = g[(name, bar, r["req"]["task"])]
                    out[i, MODEL_INDEX[name], b] = (a + 5 * ga / max(gn, 1)) / (n + 5)  # shrink to global
        return out
    return predict


def _sparse(rows):
    data, ind, ptr = [], [], [0]
    D = np.stack([dense(r["req"]) for r in rows])
    for r in rows:
        t = sorted(set(tokens(r["req"]["prompt"])))
        ind.extend(t)
        data.extend([1.0 / np.sqrt(len(t))] * len(t))
        ptr.append(len(ind))
    X = sp.csr_matrix((data, ind, ptr), shape=(len(rows), HASH_BUCKETS))
    return sp.hstack([X, sp.csr_matrix(D)]).tocsr()


def make_logreg(train_rows):
    """Linear baseline: one L2 logistic regression per (model, bar) on the same features."""
    X = _sparse(train_rows)
    fits = {}
    for m in ALL_MODELS:
        idx = [i for i, r in enumerate(train_rows) if m.name in r["outcomes"]]
        for bar in QUALITY_BARS:
            y = np.array([train_rows[i]["outcomes"][m.name][0][bar] for i in idx])
            if y.min() == y.max():
                fits[(m.name, bar)] = float(y[0])
                continue
            fits[(m.name, bar)] = LogisticRegression(C=4.0, max_iter=2000).fit(X[idx], y)

    def predict(rows):
        Xt = _sparse(rows)
        out = np.zeros((len(rows), len(ALL_MODELS), len(QUALITY_BARS)))
        for (name, bar), f in fits.items():
            j, b = MODEL_INDEX[name], BAR_IX[bar]
            out[:, j, b] = f if isinstance(f, float) else f.predict_proba(Xt)[:, 1]
        return out
    return predict


# ---------------------------------------------------------------- prediction metrics
def pairs(rows, P):
    ps, ys = [], []
    for i, r in enumerate(rows):
        b = BAR_IX[r["req"]["quality"]]
        for name, outs in r["outcomes"].items():
            ps.append(P[i, MODEL_INDEX[name], b])
            ys.append(outs[0][r["req"]["quality"]])
    return np.array(ps), np.array(ys)


def ece(p, y, bins=15):
    e = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        k = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if k.any():
            tot += k.mean() * abs(p[k].mean() - y[k].mean())
    return tot


def pred_metrics(rows, P):
    p, y = pairs(rows, P)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auc": roc_auc_score(y, p), "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), "ece": ece(p, y)}


# ---------------------------------------------------------------- routing replay
def _cands(r, P, i):
    q = req_obj(r)
    el = [BY_NAME[n] for n in r["outcomes"]]
    b = BAR_IX[q.quality]
    return q, el, [(m, float(P[i, MODEL_INDEX[m.name], b]), m.credits(q.resolution, q.seconds)) for m in el]


def replay(rows, decide, judge_acc=0.95, judge_credits=1.0, seed=0):
    """decide(i, row) -> (first_model, escalation_model | None). Every request was rendered on every
    eligible model, so each choice has a real judge outcome. For two-step plans a runtime judge (which
    agrees with the eval label with prob judge_acc) decides whether to escalate."""
    rng = np.random.default_rng(seed)
    out = []
    for i, r in enumerate(rows):
        q = req_obj(r)
        m1, esc = decide(i, r)
        ok = r["outcomes"][m1][0][q.quality]
        cost = BY_NAME[m1].credits(q.resolution, q.seconds)
        attempts = 1
        if esc is not None:
            cost += judge_credits
            verdict = ok if rng.random() < judge_acc else 1 - ok
            if not verdict:
                cost += BY_NAME[esc].credits(q.resolution, q.seconds)
                ok = r["outcomes"][esc][0][q.quality]
                attempts = 2
        out.append((m1, cost, ok, attempts))
    return out


def fixed_policy(pick):
    def decide(i, r):
        q = req_obj(r)
        return pick(q, [BY_NAME[n] for n in r["outcomes"]]).name, None
    return decide


def threshold_policy(P, tau):
    def decide(i, r):
        q, _, c = _cands(r, P, i)
        return choose(c, tau[(q.task, q.quality)]).model, None
    return decide


def lagrangian_policy(P, kappa):
    """lam = kappa * premium-default cost, so one kappa means the same trade-off for a 5-credit image
    and a 2,600-credit video."""
    def decide(i, r):
        q, el, c = _cands(r, P, i)
        prem = premium_default(q.task, el, q.resolution, q.seconds)
        return choose_lagrangian(c, kappa[(q.task, q.quality)] * prem.credits(q.resolution, q.seconds)).model, None
    return decide


def cascade_policy(P, kappa, judge_credits=1.0):
    def decide(i, r):
        q, el, c = _cands(r, P, i)
        prem = premium_default(q.task, el, q.resolution, q.seconds)
        ch, esc = choose_cascade(c, kappa[(q.task, q.quality)] * prem.credits(q.resolution, q.seconds), prem, judge_credits)
        return ch.model, esc
    return decide


def summarise(res):
    cr = np.array([x[1] for x in res])
    ok = np.array([x[2] for x in res], dtype=float)
    return {"pass_rate": float(ok.mean()), "credits_per_request": float(cr.mean()), "credits_total": float(cr.sum()),
            "escalation_rate": float(np.mean([x[3] == 2 for x in res]))}


def slices(rows):
    s = collections.defaultdict(list)
    for i, r in enumerate(rows):
        s[(r["req"]["task"], r["req"]["quality"])].append(i)
    return s


TAU_GRID = np.concatenate([np.linspace(0.5, 0.95, 19), np.linspace(0.955, 0.999, 30)])
KAPPA_GRID = np.geomspace(0.1, 3000, 60)


def tune(val_rows, make_policy, target_by_slice, grid, cheapest_first=True, **kw):
    """Per (task, bar): the grid value with the lowest val credits whose val pass rate >= the
    premium default's val pass rate in that slice. Uses the validation split only."""
    best = {}
    for key, idx in slices(val_rows).items():
        sub = [val_rows[i] for i in idx]
        choice, choice_cost = grid[-1], None
        for g in grid:
            pol = make_policy(idx, collections.defaultdict(lambda: g))
            res = replay(sub, pol, **kw)
            s = summarise(res)
            if s["pass_rate"] >= target_by_slice[key] and (choice_cost is None or s["credits_per_request"] < choice_cost):
                choice, choice_cost = g, s["credits_per_request"]
        best[key] = float(choice)
    return best


def _sub(P, idx):
    return P[idx]


def bootstrap_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [np.mean(a[i] - b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval.jsonl")
    ap.add_argument("--model", default="models/passnet.pt")
    ap.add_argument("--out", default="results/offline.json")
    ap.add_argument("--tau-out", default="models/tau.json")
    a = ap.parse_args()
    rows = load_rows(a.data)
    tr, va, te = split(rows)
    net, meta = load_bytes(pathlib.Path(a.model).read_bytes())

    preds = {}
    t0 = time.time()
    preds["passnet"] = (p_passnet(net, va), p_passnet(net, te))
    cache = pathlib.Path("data/baseline_preds.npz")
    if cache.exists():
        z = np.load(cache)
        for k in ("bucket_rules", "logreg", "oracle"):
            preds[k] = (z[k + "_v"], z[k + "_t"])
    else:
        bp = make_bucket_prior(tr)
        preds["bucket_rules"] = (bp(va), bp(te))
        lr = make_logreg(tr)
        preds["logreg"] = (lr(va), lr(te))
        preds["oracle"] = (p_oracle(va), p_oracle(te))
        np.savez(cache, **{k + s: v[j] for k, v in preds.items() if k != "passnet" for j, s in enumerate(("_v", "_t"))})
    print(f"predictions in {time.time() - t0:.0f}s")

    report = {"model_version": meta["version"], "n_test": len(te), "prediction": {}, "routing": {}}
    for k, (_, Pt) in preds.items():
        report["prediction"][k] = pred_metrics(te, Pt)
        print(f"{k:14s}", {m: round(v, 4) for m, v in report["prediction"][k].items()})

    prem = lambda q, el: premium_default(q.task, el, q.resolution, q.seconds)
    cheap = lambda q, el: min(el, key=lambda m: m.credits(q.resolution, q.seconds))
    prem_val = replay(va, fixed_policy(prem))
    target = {key: np.mean([prem_val[i][2] for i in idx]) for key, idx in slices(va).items()}

    def naive_cascade(i, r):  # no model: cheapest eligible first, premium on judge reject
        q = req_obj(r)
        el = [BY_NAME[n] for n in r["outcomes"]]
        p = prem(q, el)
        c = cheap(q, el)
        return c.name, (None if c.name == p.name else p.name)

    runs = {"premium_default": replay(te, fixed_policy(prem)), "always_cheapest": replay(te, fixed_policy(cheap)),
            "naive_cascade[cheapest->premium]": replay(te, naive_cascade)}
    params = {}
    for k, (Pv, Pt) in preds.items():
        t0 = time.time()
        tau = tune(va, lambda idx, g, Pv=Pv: threshold_policy(Pv[idx], g), target, TAU_GRID)
        runs[f"threshold[{k}]"] = replay(te, threshold_policy(Pt, tau))
        kap = tune(va, lambda idx, g, Pv=Pv: lagrangian_policy(Pv[idx], g), target, KAPPA_GRID)
        runs[f"lagrangian[{k}]"] = replay(te, lagrangian_policy(Pt, kap))
        params[k] = {"tau": tau, "kappa": kap}
        if True:
            kc = tune(va, lambda idx, g, Pv=Pv: cascade_policy(Pv[idx], g), target, KAPPA_GRID)
            runs[f"cascade[{k}]"] = replay(te, cascade_policy(Pt, kc))
            params[k]["kappa_cascade"] = kc
        print(f"tuned {k} in {time.time() - t0:.0f}s")

    base = summarise(runs["premium_default"])
    for k, res in runs.items():
        s = summarise(res)
        s["credits_saved_vs_premium"] = 1 - s["credits_per_request"] / base["credits_per_request"]
        lo, hi = bootstrap_diff([x[2] for x in res], [x[2] for x in runs["premium_default"]])
        s["pass_rate_delta_ci95"] = [lo, hi]
        per = {}
        for (task, bar), idx in slices(te).items():
            sub = [res[i] for i in idx]
            bsub = [runs["premium_default"][i] for i in idx]
            per[f"{task}/{bar}"] = {
                "n": len(idx), "pass": float(np.mean([x[2] for x in sub])),
                "premium_pass": float(np.mean([x[2] for x in bsub])),
                "credits_saved": 1 - sum(x[1] for x in sub) / sum(x[1] for x in bsub)}
        s["by_slice"] = per
        mix = collections.Counter(x[0] for x in res)
        s["model_mix"] = {m: round(c / len(res), 4) for m, c in mix.most_common()}
        report["routing"][k] = s
        print(f"{k:24s} pass {s['pass_rate']:.3f} (Δ CI [{lo:+.3f},{hi:+.3f}])  "
              f"credits/req {s['credits_per_request']:8.1f}  saved {s['credits_saved_vs_premium']:+.1%}"
              f"  esc {s['escalation_rate']:.1%}")
    # cascade sensitivity to the runtime judge's accuracy (kappa re-tuned on val for each)
    sens = {}
    for acc in (0.8, 0.9, 0.95, 1.0):
        Pv, Pt = preds["passnet"]
        kc = tune(va, lambda idx, g: cascade_policy(Pv[idx], g), target, KAPPA_GRID, judge_acc=acc)
        s = summarise(replay(te, cascade_policy(Pt, kc), judge_acc=acc))
        sens[str(acc)] = {"pass_rate": s["pass_rate"], "credits_saved_vs_premium": 1 - s["credits_per_request"] / base["credits_per_request"],
                          "escalation_rate": s["escalation_rate"]}
        print(f"cascade judge_acc={acc}: pass {s['pass_rate']:.3f} saved {sens[str(acc)]['credits_saved_vs_premium']:+.1%}")
    report["cascade_judge_sensitivity"] = sens
    report["params"] = {k: {pk: {f"{t}/{b}": v for (t, b), v in pv.items()} for pk, pv in d.items()} for k, d in params.items()}

    # cost/quality frontier on test: one global kappa swept
    curve = {}
    for k in preds:
        pts = []
        for g in np.geomspace(0.01, 1000, 50):
            s = summarise(replay(te, lagrangian_policy(preds[k][1], collections.defaultdict(lambda: g))))
            pts.append([s["credits_per_request"], s["pass_rate"]])
        curve[k] = pts
    pts = []
    for g in np.geomspace(0.01, 1000, 50):
        s = summarise(replay(te, cascade_policy(preds["passnet"][1], collections.defaultdict(lambda: g))))
        pts.append([s["credits_per_request"], s["pass_rate"]])
    curve["cascade[passnet]"] = pts
    report["pareto"] = curve
    report["premium_point"] = [base["credits_per_request"], base["pass_rate"]]

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(report, indent=1))
    tau_nested = {"image": {}, "video": {}, "kappa": {"image": {}, "video": {}}, "kappa_cascade": {"image": {}, "video": {}}}
    for (t, b), v in params["passnet"]["tau"].items():
        tau_nested[t][b] = v
    for (t, b), v in params["passnet"]["kappa"].items():
        tau_nested["kappa"][t][b] = v
    for (t, b), v in params["passnet"]["kappa_cascade"].items():
        tau_nested["kappa_cascade"][t][b] = v
    pathlib.Path(a.tau_out).write_text(json.dumps(tau_nested, indent=1))
    print(f"wrote {a.out} and {a.tau_out}")


if __name__ == "__main__":
    main()
