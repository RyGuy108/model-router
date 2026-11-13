"""FastAPI router service.

    POST /v1/route      pick a model for a request (no render)
    POST /v1/generate   route, then submit the render to Magic Hour with the chosen model
    POST /v1/feedback   report whether the render passed and what it cost
    GET  /v1/models     catalog with live costs, online bias and calibration
    GET  /healthz, /metrics

Hot path of /route: Redis score cache -> micro-batched PassNet -> live bias/cost tables (refreshed
from Redis every second, not per request) -> policy -> async batched decision log to Postgres.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import orjson
import psycopg
import redis.asyncio as aioredis
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from router.catalog import ALL_MODELS, BY_NAME, IMAGE_RES_MULT, MODEL_INDEX, QUALITY_BARS, VIDEO_RES_MULT, Request
from router.model import load_bytes
from router.policy import DEFAULT_TAU, choose, choose_cascade, choose_lagrangian, premium_default
from router.scorer import Scorer

CFG = {
    "pg": os.environ.get("ROUTER_PG", "postgresql://postgres:postgres@localhost/router"),
    "redis": os.environ.get("ROUTER_REDIS", "redis://localhost:6379/0"),
    "model_file": os.environ.get("ROUTER_MODEL_FILE", "models/passnet.pt"),
    "tau_file": os.environ.get("ROUTER_TAU_FILE", "models/tau.json"),
    "epsilon": float(os.environ.get("ROUTER_EPSILON", "0")),
    "explore": os.environ.get("ROUTER_EXPLORE", "cheaper"),  # cheaper | uniform
    "online_bias": os.environ.get("ROUTER_ONLINE_BIAS", "1") == "1",
    "bias_lr": float(os.environ.get("ROUTER_BIAS_LR", "0.02")),
    "cost_alpha": float(os.environ.get("ROUTER_COST_ALPHA", "0.1")),
    "batching": os.environ.get("ROUTER_BATCHING", "0") == "1",
    "cache_ttl": int(os.environ.get("ROUTER_CACHE_TTL", "3600")),
    "log_decisions": os.environ.get("ROUTER_LOG_DECISIONS", "1") == "1",
    "policy": os.environ.get("ROUTER_POLICY", "lagrangian"),  # lagrangian | threshold
}
BAR_IX = {b: i for i, b in enumerate(QUALITY_BARS)}
NS = os.environ.get("ROUTER_NS", "rt")  # Redis key namespace


class RouteIn(BaseModel):
    task: str = Field(pattern="^(image|video)$")
    prompt: str = Field(min_length=1, max_length=4000)
    resolution: str
    aspect_ratio: str = "16:9"
    seconds: float = 0
    has_image_input: bool = False
    quality: str = Field("standard", pattern="^(draft|standard|premium)$")
    allowed_models: list[str] | None = None
    max_credits: float | None = None
    mode: str = Field("single", pattern="^(single|cascade)$")  # cascade: you run a judge and escalate on reject
    tenant: str = "default"


class FeedbackIn(BaseModel):
    decision_id: uuid.UUID
    passed: bool
    credits_charged: float | None = None
    model: str | None = None  # defaults to the routed model


class State:
    scorer: Scorer
    tau: dict
    pg: AsyncConnectionPool
    r: aioredis.Redis
    bias = np.zeros(len(ALL_MODELS))
    cost_scale = np.ones(len(ALL_MODELS))  # observed / catalog cost, EMA
    log_q: asyncio.Queue
    lat: list[float] = []
    counters: dict = {"route": 0, "cache_hit": 0, "explore": 0, "fallback": 0, "feedback": 0}


S = State()

# atomic EMA update shared by all API workers
EMA_LUA = """
local old = redis.call('HGET', KEYS[1], ARGV[1])
local obs, a = tonumber(ARGV[2]), tonumber(ARGV[3])
local new = obs
if old then new = (1 - a) * tonumber(old) + a * obs end
redis.call('HSET', KEYS[1], ARGV[1], new)
return tostring(new)
"""


def units(spec, resolution, seconds):
    return IMAGE_RES_MULT[resolution] if spec.task == "image" else seconds * VIDEO_RES_MULT[resolution]


# ------------------------------------------------------------------ model registry
async def load_active_model():
    async with S.pg.connection() as c:
        row = await (await c.execute("SELECT version, artifact, tau FROM model_registry WHERE active")).fetchone()
        if row is None and os.path.exists(CFG["model_file"]):
            blob = open(CFG["model_file"], "rb").read()
            _, meta = load_bytes(blob)
            tau = json.load(open(CFG["tau_file"])) if os.path.exists(CFG["tau_file"]) else DEFAULT_TAU
            await c.execute("INSERT INTO model_registry (version, artifact, tau, active) VALUES (%s,%s,%s,true) "
                            "ON CONFLICT (version) DO NOTHING", (meta["version"], blob, json.dumps(tau)))
            row = (meta["version"], blob, tau)
    if row is None:
        raise RuntimeError("no active model in registry and no model file")
    net, meta = load_bytes(bytes(row[1]))
    return net, row[0], row[2]


async def watch_model_updates():
    """Hot reload: `python -m router.registry activate` publishes the new version."""
    ps = S.r.pubsub()
    await ps.subscribe(f"{NS}:model")
    async for msg in ps.listen():
        if msg["type"] != "message":
            continue
        net, version, tau = await load_active_model()
        S.scorer.swap(net, version)
        S.tau = tau


async def refresh_tables():
    """Online bias and live cost tables live in Redis (shared by all workers); each worker
    snapshots them once a second so the hot path never waits on Redis for them."""
    while True:
        try:
            b = await S.r.hgetall(f"{NS}:bias")
            c = await S.r.hgetall(f"{NS}:cost_scale")
            bias = np.zeros(len(ALL_MODELS))
            cs = np.ones(len(ALL_MODELS))
            for k, v in b.items():
                if k.decode() in MODEL_INDEX:
                    bias[MODEL_INDEX[k.decode()]] = float(v)
            for k, v in c.items():
                if k.decode() in MODEL_INDEX:
                    cs[MODEL_INDEX[k.decode()]] = float(v)
            S.bias = np.clip(bias, -3, 3) if CFG["online_bias"] else np.zeros(len(ALL_MODELS))
            S.cost_scale = cs
        except Exception:
            pass
        await asyncio.sleep(1.0)


async def decision_writer():
    """Batch decision rows into one multi-row INSERT per ~20 ms instead of one transaction per request."""
    sql = ("INSERT INTO decisions (id, tenant, model_version, request, chosen, p_pass, est_credits, propensity, "
           "explored, reason, scores) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING")
    while True:
        rows = [await S.log_q.get()]
        await asyncio.sleep(0.02)
        while not S.log_q.empty() and len(rows) < 2000:
            rows.append(S.log_q.get_nowait())
        for attempt in range(5):
            try:
                async with S.pg.connection() as c:
                    async with c.cursor() as cur:
                        await cur.executemany(sql, rows)
                break
            except psycopg.OperationalError:
                await asyncio.sleep(0.2 * 2 ** attempt)


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.environ.get("ROUTER_TORCH_THREADS", "1")))
    S.pg = AsyncConnectionPool(CFG["pg"], min_size=2, max_size=8, open=False)
    await S.pg.open()
    async with S.pg.connection() as c:
        await c.execute(open(os.path.join(os.path.dirname(__file__), "schema.sql")).read())
    S.r = aioredis.from_url(CFG["redis"])
    net, version, tau = await load_active_model()
    S.tau = tau
    S.scorer = Scorer(net, version, batching=CFG["batching"])
    await S.scorer.start()
    S.log_q = asyncio.Queue()
    tasks = [asyncio.create_task(t()) for t in (watch_model_updates, refresh_tables, decision_writer)]
    yield
    while not S.log_q.empty():  # flush decision log
        await asyncio.sleep(0.05)
    for t in tasks:
        t.cancel()
    await S.pg.close()
    await S.r.aclose()


app = FastAPI(title="Magic Hour model router", lifespan=lifespan)


def cache_key(body: RouteIn, version: str) -> str:
    k = json.dumps([body.task, body.prompt.strip().lower(), body.resolution, body.aspect_ratio,
                    body.seconds, body.has_image_input], separators=(",", ":"))
    return f"{NS}:score:{version}:" + hashlib.blake2b(k.encode(), digest_size=16).hexdigest()


async def get_scores(body: RouteIn) -> tuple[np.ndarray, str, bool]:
    key = cache_key(body, S.scorer.version)
    raw = await S.r.get(key)
    if raw is not None:
        return np.frombuffer(raw, dtype=np.float32).reshape(len(ALL_MODELS), len(QUALITY_BARS)), S.scorer.version, True
    z, version = await S.scorer.score(body.model_dump())
    await S.r.set(cache_key(body, version), z.astype(np.float32).tobytes(), ex=CFG["cache_ttl"])
    return z, version, False


async def route(body: RouteIn) -> dict:
    t0 = time.perf_counter()
    req = Request(task=body.task, prompt=body.prompt, resolution=body.resolution, aspect_ratio=body.aspect_ratio,
                  seconds=body.seconds, has_image_input=body.has_image_input, quality=body.quality,
                  allowed_models=body.allowed_models, max_credits=body.max_credits)
    eligible = req.eligible()
    if not eligible:
        raise HTTPException(422, "no model supports this task/resolution/duration")
    z, version, hit = await get_scores(body)
    b = BAR_IX[body.quality]
    cands = []
    for m in eligible:
        j = MODEL_INDEX[m.name]
        p = 1 / (1 + math.exp(-(float(z[j, b]) + S.bias[j])))
        cands.append((m, p, m.credits(body.resolution, body.seconds) * S.cost_scale[j]))
    tau = S.tau.get(body.task, {}).get(body.quality, 0.9)
    prem = premium_default(body.task, eligible, body.resolution, body.seconds)
    prem_cost = cands[[x[0].name for x in cands].index(prem.name)][2]
    escalate_to = None
    if body.mode == "cascade" and body.max_credits is None:
        kappa = S.tau.get("kappa_cascade", {}).get(body.task, {}).get(body.quality, 1.0)
        c, escalate_to = choose_cascade(cands, kappa * prem_cost, prem)
    elif CFG["policy"] == "lagrangian" and "kappa" in S.tau:
        kappa = S.tau["kappa"].get(body.task, {}).get(body.quality, 1.0)
        c = choose_lagrangian(cands, kappa * prem_cost, body.max_credits)
    else:
        c = choose(cands, tau, body.max_credits)
    chosen, explored, propensity = c, False, 1.0
    eps = CFG["epsilon"]
    if eps > 0:
        # cheaper: explore only models cheaper than the pick (where savings hide).
        # uniform: explore any eligible model, which gives every model a logged propensity > 0,
        # so any future policy can be evaluated off-policy from these logs (bench/ope.py).
        pool = [x for x in cands if x[2] < c.credits] if CFG["explore"] == "cheaper" else cands
        if pool:
            k = len(pool)
            in_pool = any(x[0].name == c.model for x in pool)
            propensity = (1 - eps) + (eps / k if in_pool else 0.0)
            if random.random() < eps:
                m, p, cr = random.choice(pool)
                if m.name != c.model:
                    chosen = type(c)(m.name, p, cr, "explore")
                    explored, propensity = True, eps / k
    did = uuid.uuid4()
    scores = {x[0].name: {"p": round(x[1], 5), "c": round(x[2], 3)} for x in cands}
    out = {
        "decision_id": str(did), "model": chosen.model, "p_pass": round(chosen.p_pass, 4),
        "est_credits": round(chosen.credits, 2), "reason": chosen.reason, "explored": explored,
        "escalate_to": escalate_to if not explored else prem.name if body.mode == "cascade" else None,
        "model_version": version, "cache_hit": hit,
        "alternatives": sorted(({"model": x[0].name, "p_pass": round(x[1], 4), "est_credits": round(x[2], 2)}
                                for x in cands), key=lambda a: a["est_credits"]),
    }
    # remember what we predicted so /feedback can update online stats without touching Postgres
    await S.r.set(f"{NS}:dec:{did}", json.dumps([chosen.model, chosen.p_pass, body.resolution, body.seconds]), ex=86400 * 3)
    if CFG["log_decisions"]:
        S.log_q.put_nowait((did, body.tenant, version, json.dumps(body.model_dump()), chosen.model, chosen.p_pass,
                            chosen.credits, propensity, explored, chosen.reason, json.dumps(scores)))
    S.counters["route"] += 1
    S.counters["cache_hit"] += hit
    S.counters["explore"] += explored
    S.counters["fallback"] += chosen.reason == "fallback_best"
    S.lat.append(time.perf_counter() - t0)
    if len(S.lat) > 50000:
        S.lat = S.lat[-20000:]
    return out


@app.post("/v1/route")
async def route_ep(body: RouteIn):
    return Response(orjson.dumps(await route(body), option=orjson.OPT_SERIALIZE_NUMPY), media_type="application/json")


@app.post("/v1/feedback")
async def feedback(fb: FeedbackIn):
    raw = await S.r.get(f"{NS}:dec:{fb.decision_id}")
    if raw is None:
        raise HTTPException(404, "unknown or expired decision_id")
    model, p, res, secs = json.loads(raw)
    model = fb.model or model
    async with S.pg.connection() as c:
        cur = await c.execute("INSERT INTO feedback (decision_id, model, passed, credits_charged) VALUES (%s,%s,%s,%s) "
                              "ON CONFLICT DO NOTHING", (fb.decision_id, model, fb.passed, fb.credits_charged))
        first = cur.rowcount == 1
    if first:  # idempotent: a retried feedback call does not double-update online state
        y = 1.0 if fb.passed else 0.0
        pipe = S.r.pipeline()
        # online intercept recalibration: one SGD step on the per-model logit offset (logistic loss)
        pipe.hincrbyfloat(f"{NS}:bias", model, CFG["bias_lr"] * (y - p))
        pipe.hincrbyfloat(f"{NS}:calib:n", model, 1)
        pipe.hincrbyfloat(f"{NS}:calib:p", model, p)
        pipe.hincrbyfloat(f"{NS}:calib:y", model, y)
        await pipe.execute()
        if fb.credits_charged is not None and model in BY_NAME:
            spec = BY_NAME[model]
            obs = fb.credits_charged / max(1e-9, spec.credits(res, secs))
            await S.r.eval(EMA_LUA, 1, f"{NS}:cost_scale", model, obs, CFG["cost_alpha"])
        S.counters["feedback"] += 1
    return {"ok": True, "applied": first}


@app.post("/v1/generate")
async def generate(body: RouteIn):
    """Route, then submit to the real Magic Hour API (images only in this version)."""
    if body.task != "image":
        raise HTTPException(501, "generate supports task=image; use /v1/route for video")
    d = await route(body)
    from magic_hour import Client
    client = Client(token=os.environ["MAGIC_HOUR_API_KEY"])
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, lambda: client.v1.ai_image_generator.create(
        image_count=1, style={"prompt": body.prompt}, model=d["model"], resolution=body.resolution,
        aspect_ratio=body.aspect_ratio, name=f"router-{d['decision_id'][:8]}"))
    d["project_id"] = res.id
    d["credits_charged"] = getattr(res, "credits_charged", None)
    return d


@app.get("/v1/models")
async def models():
    n = await S.r.hgetall(f"{NS}:calib:n")
    p = await S.r.hgetall(f"{NS}:calib:p")
    y = await S.r.hgetall(f"{NS}:calib:y")
    out = []
    for m in ALL_MODELS:
        k = m.name.encode()
        cnt = float(n.get(k, 0))
        out.append({"model": m.name, "task": m.task, "catalog_cost": m.base_cost, "cost_source": m.cost_source,
                    "cost_scale": float(S.cost_scale[MODEL_INDEX[m.name]]), "bias": float(S.bias[MODEL_INDEX[m.name]]),
                    "feedback_n": int(cnt),
                    "mean_predicted": float(p.get(k, 0)) / cnt if cnt else None,
                    "mean_observed": float(y.get(k, 0)) / cnt if cnt else None})
    return {"model_version": S.scorer.version, "tau": S.tau, "models": out}


@app.get("/healthz")
async def healthz():
    await S.r.ping()
    return {"ok": True, "model_version": S.scorer.version}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    lat = np.array(S.lat[-20000:]) * 1000 if S.lat else np.zeros(1)
    lines = [f"router_{k}_total {v}" for k, v in S.counters.items()]
    lines += [f'router_route_latency_ms{{q="{q}"}} {np.percentile(lat, q * 100):.3f}' for q in (0.5, 0.99)]
    lines.append(f"router_scorer_batches_total {S.scorer.batches}")
    lines.append(f"router_scorer_items_total {S.scorer.items}")
    lines.append(f"router_decision_log_backlog {S.log_q.qsize()}")
    return "\n".join(lines) + "\n"
