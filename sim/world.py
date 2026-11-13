"""Simulated eval harness.

There is no real eval harness with per-model human/judge labels yet, so this module stands in for one.
It generates realistic API requests from a prompt grammar, and for every (request, model) pair it
samples a quality score and the judge's pass/fail at each quality bar, the same shape of data an eval
harness that renders every prompt on every model would produce.

How quality is generated (all of this is hidden from the router):
  * every prompt fragment carries a "demand" vector over 10 capabilities (text rendering,
    photorealism, anatomy, composition, fine detail, stylization, motion, physics, temporal
    consistency, camera work). Resolution and duration add to detail/temporal demand.
  * every model has a hidden skill vector over the same capabilities and a base quality.
    THE SKILL PROFILES ARE INVENTED FOR THE SIMULATION. They are not measurements of the real
    Magic Hour models and must not be quoted as such.
  * gap_k = w_k * demand_k * (1 - skill_k); quality = base - softmax_k(gap_k) - 0.12 * sum_k gap_k
    + prompt_difficulty + render_noise  (a render fails on its worst defect, so capabilities interact)
    prompt_difficulty is shared across models and NOT visible in the prompt text, and render_noise
    is per render, so there is irreducible uncertainty that no router can learn away.
  * the judge passes a render at bar b if quality + judge_noise > threshold[b].

The router only sees the request (prompt text, task, resolution, seconds, aspect, bar).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

import numpy as np

from router.catalog import ALL_MODELS, MODELS_FOR_TASK, Request

DIMS = ("text", "photo", "anatomy", "compose", "detail", "style", "motion", "physics", "temporal", "camera")
D = {k: i for i, k in enumerate(DIMS)}
W = np.array([1.3, 0.8, 1.1, 1.0, 0.7, 0.6, 1.1, 1.0, 1.0, 0.7])
BETA, GAP_MAX, GAP_SUM = 8.0, 1.0, 0.12
THRESH = {"draft": 0.20, "standard": 0.40, "premium": 0.55}


def dv(_fill: float = 0.0, **kw) -> np.ndarray:
    v = np.full(len(DIMS), _fill)
    for k, x in kw.items():
        v[D[k]] = x
    return v


# --- hidden model profiles (invented) -------------------------------------------------------
SKILL = {
    # image
    "z-image-turbo":   (0.70, dv(.5, text=.15, photo=.70, anatomy=.45, compose=.30, detail=.35, style=.50)),
    "flux-schnell":    (0.70, dv(.5, text=.30, photo=.50, anatomy=.45, compose=.40, detail=.40, style=.70)),
    "qwen-image-2.1":  (0.70, dv(.5, text=.85, photo=.45, anatomy=.50, compose=.55, detail=.40, style=.55)),
    "krea-2":          (0.73, dv(.5, text=.25, photo=.85, anatomy=.60, compose=.40, detail=.55, style=.45)),
    "seedream-v4":     (0.75, dv(.5, text=.60, photo=.75, anatomy=.70, compose=.65, detail=.70, style=.70)),
    "gpt-image-2":     (0.77, dv(.5, text=.92, photo=.65, anatomy=.75, compose=.90, detail=.60, style=.80)),
    "seedream-v5-pro": (0.79, dv(.5, text=.70, photo=.85, anatomy=.80, compose=.75, detail=.90, style=.75)),
    "nano-banana-pro": (0.83, dv(.5, text=.92, photo=.88, anatomy=.88, compose=.90, detail=.85, style=.85)),
    # video
    "kling-2.6":         (0.82, dv(.5, photo=.55, anatomy=.45, motion=.70, physics=.40, temporal=.35, camera=.75, style=.45)),
    "ltx-2.5":           (0.80, dv(.5, photo=.50, anatomy=.40, motion=.35, physics=.35, temporal=.60, camera=.45, style=.55)),
    "kling-3.0":         (0.87, dv(.5, photo=.70, anatomy=.65, motion=.75, physics=.55, temporal=.60, camera=.80, style=.60, text=.30)),
    "seedance-2.0-mini": (0.87, dv(.5, photo=.65, anatomy=.65, motion=.65, physics=.60, temporal=.70, camera=.60, style=.70, text=.30)),
    "veo3.1-lite":       (0.89, dv(.5, photo=.85, anatomy=.75, motion=.65, physics=.70, temporal=.65, camera=.65, style=.50, text=.45)),
    "seedance-2.0":      (0.95, dv(.5, photo=.90, anatomy=.88, motion=.88, physics=.85, temporal=.85, camera=.85, style=.85, text=.55)),
}

# --- prompt grammar ---------------------------------------------------------------------------
# Each fragment: (surface forms, demand vector). Several surface forms per fragment so the model
# has to learn from words rather than memorise whole templates.
SUBJECTS = [
    (["a golden retriever puppy", "a fluffy dog", "a young labrador"], dv(anatomy=.1)),
    (["a red vintage convertible", "an old red sports car", "a classic cherry-red car"], dv(detail=.2)),
    (["a steaming bowl of ramen", "a bowl of noodle soup", "hot ramen in a ceramic bowl"], dv(detail=.3, photo=.2)),
    (["a mountain lake at dawn", "an alpine lake at sunrise", "a misty mountain lake"], dv()),
    (["a modern glass skyscraper", "a tall office tower", "a futuristic high-rise"], dv(detail=.3)),
    (["a woman laughing", "a smiling woman", "a woman with a big laugh"], dv(anatomy=.55, photo=.3)),
    (["an elderly man reading", "a grandfather reading a newspaper", "an old man with a book"], dv(anatomy=.5, photo=.2)),
    (["two chefs high-fiving", "a pair of cooks giving each other a high five"], dv(anatomy=.8, compose=.55)),
    (["a crowd of marathon runners", "dozens of runners at a race start", "a packed marathon start line"], dv(anatomy=.7, compose=.85, detail=.4)),
    (["a hand holding a smartphone", "close-up of fingers typing on a phone", "a hand gripping a coffee mug"], dv(anatomy=.85, detail=.3)),
    (["a family of four at dinner", "four friends around a table", "a group of coworkers in a meeting"], dv(anatomy=.65, compose=.7)),
    (["an astronaut cat", "a cat in a space suit", "a kitten astronaut"], dv(style=.2, compose=.2)),
    (["a dragon over a castle", "a huge dragon circling a fortress"], dv(style=.4, compose=.4, detail=.3)),
    (["a glass chess set", "crystal chess pieces on a board"], dv(detail=.6, compose=.4, photo=.3)),
    (["a sneaker product shot", "a running shoe on a white background", "a product photo of headphones"], dv(photo=.5, detail=.3)),
    (["a robot barista", "a humanoid robot making coffee"], dv(compose=.3, detail=.3, anatomy=.3)),
    (["a tiny house in a forest", "a cabin in the woods"], dv()),
    (["a bowl of fruit", "a plate of oranges", "a basket of apples"], dv(photo=.2)),
]

TEXT_FRAGMENTS = [  # text rendering demand scales with the number of quoted words
    ("with a neon sign that reads \"{t}\"", .55),
    ("next to a storefront banner saying \"{t}\"", .55),
    ("on a book cover titled \"{t}\"", .6),
    ("with the words \"{t}\" printed on a t-shirt", .5),
    ("as a poster with the headline \"{t}\"", .65),
    ("with a chalkboard menu listing \"{t}\"", .8),
]
TEXT_SNIPPETS = [
    "OPEN", "SALE", "Grand Opening", "Fresh Coffee Daily", "The Last Summer", "Hello World",
    "Soup of the Day: Tomato Basil", "Buy One Get One Free This Weekend", "MAGIC", "Closing Sale 50% Off Everything",
    "Night Market", "Est. 1998", "Welcome to Austin Texas", "Espresso 3 Latte 4 Mocha 5 Tea 2",
]

COUNTING = [
    (["with exactly three apples on the table", "with precisely five balloons"], dv(compose=.7)),
    (["standing to the left of a blue door", "behind a red bicycle and in front of a fence"], dv(compose=.55)),
]

STYLES = [
    (["photorealistic, 85mm lens", "ultra realistic photo", "shot on a DSLR, natural light"], dv(photo=.75)),
    (["cinematic film still", "movie still, anamorphic"], dv(photo=.55, style=.15)),
    (["watercolor illustration", "soft watercolor painting"], dv(style=.55)),
    (["anime style", "anime key visual"], dv(style=.5)),
    (["flat vector illustration", "minimal flat vector art"], dv(style=.35)),
    (["pixel art", "16-bit pixel art"], dv(style=.5)),
    (["isometric 3d render", "isometric low-poly 3d"], dv(style=.45, compose=.2)),
    (["oil painting", "impressionist oil painting"], dv(style=.45)),
    ([""], dv()),
]

DETAIL_MODS = [
    (["intricate details", "highly detailed"], dv(detail=.4)),
    (["macro shot", "extreme close-up"], dv(detail=.55, photo=.2)),
    ([""], dv()),
    ([""], dv()),
]

ACTIONS = [  # video only
    (["sitting still", "standing calmly", "resting quietly"], dv(motion=.05)),
    (["slowly turning its head", "blinking and breathing gently", "with a soft breeze"], dv(motion=.12)),
    (["as clouds drift by in a time-lapse", "as the light slowly changes"], dv(motion=.1, temporal=.1)),
    (["walking slowly", "strolling down a street"], dv(motion=.25, anatomy=.15)),
    (["running fast", "sprinting"], dv(motion=.55, anatomy=.3)),
    (["doing a backflip", "performing a gymnastics flip"], dv(motion=.95, anatomy=.6, physics=.5)),
    (["dancing", "breakdancing"], dv(motion=.8, anatomy=.55)),
    (["pouring coffee into a cup", "pouring water into a glass"], dv(physics=.85, motion=.3)),
    (["as waves crash around it", "in heavy rain"], dv(physics=.65, detail=.2)),
    (["juggling three balls", "catching a frisbee"], dv(physics=.7, motion=.6, anatomy=.4)),
    (["with leaves blowing in the wind", "with smoke drifting past"], dv(physics=.4)),
    (["turning to face the camera and smiling", "looking up and waving"], dv(anatomy=.45, motion=.3, temporal=.3)),
]
CAMERAS = [
    (["static shot", "locked-off camera"], dv()),
    (["slow dolly in", "gentle push in"], dv(camera=.3)),
    (["drone shot orbiting the subject", "aerial orbit"], dv(camera=.75)),
    (["handheld tracking shot", "camera follows from behind"], dv(camera=.6, temporal=.2)),
    (["fast whip pan", "crash zoom"], dv(camera=.85, motion=.3)),
    ([""], dv()),
]


@dataclass
class Example:
    req: dict
    demand: list
    difficulty: float
    template: str


def _pick(rng: random.Random, frags):
    forms, d = rng.choice(frags)
    return rng.choice(forms), d


def sample_request(rng: random.Random) -> tuple[Request, np.ndarray, str]:
    task = "image" if rng.random() < 0.65 else "video"
    parts, demand, tags = [], np.zeros(len(DIMS)), []

    # Production traffic has a heavy head of simple requests (one subject, a style, a gentle shot).
    simple = rng.random() < 0.4
    s, d = _pick(rng, SUBJECTS); parts.append(s); demand += d; tags.append(s[:12])
    if task == "video":
        s, d = _pick(rng, ACTIONS[:3] if simple else ACTIONS); parts.append(s); demand += d
    if not simple and rng.random() < (0.45 if task == "image" else 0.12):
        frag, base = rng.choice(TEXT_FRAGMENTS)
        t = rng.choice(TEXT_SNIPPETS)
        parts.append(frag.format(t=t))
        demand += dv(text=min(1.4, base + 0.12 * len(t.split())))
    if not simple and rng.random() < 0.25:
        s, d = _pick(rng, COUNTING); parts.append(s); demand += d
    s, d = _pick(rng, STYLES)
    if s:
        parts.append(s)
    demand += d
    if not simple:
        s, d = _pick(rng, DETAIL_MODS)
        if s:
            parts.append(s)
        demand += d
    if task == "video":
        s, d = _pick(rng, CAMERAS[:2] + CAMERAS[-1:] if simple else CAMERAS)
        if s:
            parts.append(s)
        demand += d

    prompt = ", ".join(parts)
    if task == "image":
        res = rng.choices(["640px", "1k", "2k", "4k"], weights=[0.25, 0.45, 0.22, 0.08])[0]
        seconds = 0.0
        demand += dv(detail={"640px": 0, "1k": .05, "2k": .25, "4k": .5}[res])
    else:
        res = rng.choices(["480p", "720p", "1080p", "4k"], weights=[0.15, 0.5, 0.3, 0.05])[0]
        seconds = 5.0 if simple else float(rng.choices([4, 5, 8, 10, 15, 30], weights=[.1, .4, .2, .15, .1, .05])[0])
        demand += dv(detail={"480p": 0, "720p": .05, "1080p": .2, "4k": .45}[res],
                     temporal=max(0.0, (seconds - 5) / 10) * 0.6)
    quality = rng.choices(["draft", "standard", "premium"], weights=[0.25, 0.5, 0.25])[0]
    req = Request(task=task, prompt=prompt, resolution=res, seconds=seconds,
                  aspect_ratio=rng.choice(["1:1", "16:9", "9:16"]), quality=quality,
                  has_image_input=(task == "video" and rng.random() < 0.4))
    if req.has_image_input:  # a start frame pins down appearance; motion is still hard
        demand = demand * np.where(np.isin(np.arange(len(DIMS)), [D["photo"], D["style"], D["detail"]]), 0.5, 1.0)
    return req, demand, "|".join(tags)


@dataclass
class ModelDrift:
    """A change to a model's hidden quality after release (for drift experiments)."""
    model: str
    base_delta: float = 0.0
    skill_delta: dict | None = None


def quality(model: str, demand: np.ndarray, difficulty: float, noise: float,
            drift: dict[str, ModelDrift] | None = None) -> float:
    base, skill = SKILL[model]
    if drift and model in drift:
        dr = drift[model]
        base = base + dr.base_delta
        if dr.skill_delta:
            skill = skill.copy()
            for k, v in dr.skill_delta.items():
                skill[D[k]] = min(1.0, skill[D[k]] + v)
    g = W * demand * (1.0 - skill)
    # A render is usually rejected for its single worst defect (garbled text OR a sixth finger), so
    # the gap is a soft maximum over capabilities plus a small additive term for accumulated issues.
    worst = (np.log(np.sum(np.exp(BETA * g))) - np.log(len(g))) / BETA
    gap = GAP_MAX * worst + GAP_SUM * float(np.sum(g))
    return float(base - gap + difficulty + noise)


def judge(q: float, rng: np.random.Generator) -> dict[str, int]:
    j = q + rng.normal(0, 0.06)
    return {b: int(j > t) for b, t in THRESH.items()}


def build_dataset(n: int, seed: int, renders_per_pair: int = 1) -> list[dict]:
    """Every request rendered on every eligible model, like an eval harness sweep."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        req, demand, tmpl = sample_request(rng)
        difficulty = float(nrng.normal(0, 0.12))
        outcomes = {}
        for m in MODELS_FOR_TASK[req.task]:
            if not m.supports(req.resolution, req.seconds):
                continue
            outs = []
            for _ in range(renders_per_pair):
                q = quality(m.name, demand, difficulty, float(nrng.normal(0, 0.09)))
                outs.append({"q": round(q, 4), **judge(q, nrng)})
            outcomes[m.name] = outs
        if not outcomes:
            continue
        rows.append({"id": i, "req": asdict(req), "demand": demand.round(3).tolist(),
                     "difficulty": round(difficulty, 4), "template": tmpl, "outcomes": outcomes})
    return rows


def main():
    import argparse
    import pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/eval.jsonl")
    a = ap.parse_args()
    rows = build_dataset(a.n, a.seed)
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} requests, {sum(len(r['outcomes']) for r in rows)} renders -> {p}")


if __name__ == "__main__":
    main()


SIGMA_TOTAL = float(np.sqrt(0.12 ** 2 + 0.09 ** 2 + 0.06 ** 2))  # difficulty + render + judge noise


def true_pass_prob(model: str, demand, bar: str, drift: dict[str, ModelDrift] | None = None) -> float:
    """Bayes-optimal P(pass) for someone who knows the hidden model profiles and the prompt's demand
    vector but not the per-prompt difficulty or the render's dice roll. An upper bound for any router."""
    from math import erf, sqrt
    mu = quality(model, np.asarray(demand), 0.0, 0.0, drift)
    z = (mu - THRESH[bar]) / SIGMA_TOTAL
    return 0.5 * (1 + erf(z / sqrt(2)))
