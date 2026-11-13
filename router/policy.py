"""Routing policy: given calibrated P(pass) for each eligible model, pick one.

Default policy ("cheapest-confident"): the cheapest eligible model whose calibrated pass
probability at the requested bar is >= tau[task][bar]; if none clears it, the model with the
highest pass probability (ties -> cheaper). tau is tuned offline so the routed pass rate matches
the premium-default pass rate on held-out data ("equal quality"), see bench/offline.py.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from router.catalog import BY_NAME, ModelSpec

DEFAULT_TAU = {
    "image": {"draft": 0.9, "standard": 0.9, "premium": 0.9},
    "video": {"draft": 0.9, "standard": 0.9, "premium": 0.9},
}


@dataclass
class Choice:
    model: str
    p_pass: float
    credits: float
    reason: str


def choose(cands: list[tuple[ModelSpec, float, float]], tau: float, max_credits: float | None = None) -> Choice:
    """cands: (spec, p_pass, credits). Returns the routing decision."""
    if max_credits is not None:
        within = [c for c in cands if c[2] <= max_credits]
        cands = within or [min(cands, key=lambda c: c[2])]
    ok = [c for c in cands if c[1] >= tau]
    if ok:
        spec, p, cr = min(ok, key=lambda c: (c[2], -c[1]))
        return Choice(spec.name, p, cr, "cheapest_confident")
    spec, p, cr = max(cands, key=lambda c: (c[1], -c[2]))
    return Choice(spec.name, p, cr, "fallback_best")


def load_tau(path: str | pathlib.Path | None) -> dict:
    if path and pathlib.Path(path).exists():
        return json.loads(pathlib.Path(path).read_text())
    return DEFAULT_TAU


def premium_default(task: str, eligible: list[ModelSpec], resolution: str, seconds: float) -> ModelSpec:
    from router.catalog import PREMIUM_DEFAULT
    names = {m.name for m in eligible}
    if PREMIUM_DEFAULT[task] in names:
        return BY_NAME[PREMIUM_DEFAULT[task]]
    return max(eligible, key=lambda m: m.credits(resolution, seconds))


def choose_lagrangian(cands: list[tuple[ModelSpec, float, float]], lam: float,
                      max_credits: float | None = None) -> Choice:
    """argmin_m credits_m - lam * p_m: spend credits only where a model's extra pass probability
    is worth at least 1/lam credits per point. lam is tuned per (task, bar) to hit the quality target.
    Unlike a fixed threshold, this compares models by marginal value, so a 5-credit model at p=.93
    can beat a 150-credit one at p=.95 while a 2,600-credit video is only bought when it matters."""
    if max_credits is not None:
        within = [c for c in cands if c[2] <= max_credits]
        cands = within or [min(cands, key=lambda c: c[2])]
    spec, p, cr = min(cands, key=lambda c: (c[2] - lam * c[1], c[2]))
    return Choice(spec.name, p, cr, "lagrangian")


def choose_cascade(cands: list[tuple[ModelSpec, float, float]], lam: float, premium: ModelSpec,
                   judge_credits: float = 1.0) -> tuple[Choice, str | None]:
    """Two-step plan: render on m1; if an automatic judge rejects it, re-render on the premium model.
    Minimises expected credits c1 + judge + (1-p1) * c_prem - lam * (p1 + (1-p1) * p_prem).
    Returns the first choice and the escalation model (None when m1 is already the premium)."""
    by = {c[0].name: c for c in cands}
    _, pp, cp = by[premium.name]
    best, best_v = None, None
    for spec, p, cr in cands:
        if spec.name == premium.name:
            v = cp - lam * pp
        else:
            v = cr + judge_credits + (1 - p) * cp - lam * (p + (1 - p) * pp)
        if best_v is None or v < best_v:
            best, best_v = (spec, p, cr), v
    spec, p, cr = best
    esc = None if spec.name == premium.name else premium.name
    return Choice(spec.name, p, cr, "cascade"), esc
