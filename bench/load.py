"""Closed-loop load test for /v1/route.

    python -m bench.load --url http://localhost:8000 --seconds 20 --procs 4 --conc 32 [--repeat 0.3]

Requests are drawn from the held-out test split. --repeat is the fraction of requests that reuse a
recent prompt (retries / templated apps), which exercises the Redis score cache.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import random
import time

import httpx  # only for the /metrics scrape
import numpy as np

from router.train import load_rows, split


def bodies(n_unique: int, seed: int):
    rows = load_rows("data/eval.jsonl")
    _, _, te = split(rows)
    rng = random.Random(seed)
    out = []
    for r in rng.sample(te, min(n_unique, len(te))):
        q = dict(r["req"])
        q.pop("extra", None)
        q.pop("allowed_models", None)
        q.pop("max_credits", None)
        # make every non-repeat request unique so it misses the cache
        q["prompt"] = q["prompt"] + f" #{rng.randrange(10**9)}"
        out.append(q)
    return out


async def worker(url, reqs, deadline, repeat, lat, errs, rng):
    """Minimal keep-alive HTTP/1.1 client on raw asyncio streams: httpx costs more CPU per request
    than the server does, which on a shared 4-core box would measure the client, not the router."""
    host, port = url.split("//")[1].split(":")
    reader, writer = await asyncio.open_connection(host, int(port))
    recent = []
    i = 0
    while time.perf_counter() < deadline:
        if recent and rng.random() < repeat:
            payload = rng.choice(recent)
        else:
            body = dict(reqs[i % len(reqs)])
            body["prompt"] += f" {rng.randrange(10**9)}"
            i += 1
            payload = json.dumps(body).encode()
            recent = (recent + [payload])[-50:]
        msg = (b"POST /v1/route HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
               + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        t0 = time.perf_counter()
        writer.write(msg)
        status = await asyncio.wait_for(reader.readline(), 10)
        clen = 0
        while True:
            h = await reader.readline()
            if h in (b"\r\n", b""):
                break
            if h[:15].lower() == b"content-length:":
                clen = int(h[15:])
        await reader.readexactly(clen)
        lat.append(time.perf_counter() - t0)
        if b" 200 " not in status:
            errs.append(status)
    writer.close()


def proc_main(url, seconds, conc, repeat, seed, q):
    reqs = bodies(4000, seed)
    salt = f" run{time.time_ns() % 10**9}"  # never reuse cache entries from an earlier run
    for b in reqs:
        b["prompt"] += salt
    rng = random.Random(seed)
    lat, errs = [], []

    async def go():
        deadline = time.perf_counter() + seconds
        await asyncio.gather(*(worker(url, reqs[k::conc], deadline, repeat, lat, errs, random.Random(seed * 1000 + k))
                               for k in range(conc)))
    asyncio.run(go())
    q.put((lat, errs))


def run(url, seconds, procs, conc, repeat):
    q = mp.Queue()
    ps = [mp.Process(target=proc_main, args=(url, seconds, conc, repeat, s + 1, q)) for s in range(procs)]
    t0 = time.perf_counter()
    for p in ps:
        p.start()
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()
    wall = time.perf_counter() - t0
    lat = np.concatenate([np.array(r[0]) for r in res]) * 1000
    errs = sum(len(r[1]) for r in res)
    m = httpx.get(url + "/metrics").text
    return {"requests": int(len(lat)), "errors": int(errs), "rps": len(lat) / seconds,
            "p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99)),
            "p999_ms": float(np.percentile(lat, 99.9)), "concurrency": procs * conc, "server_metrics": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--repeat", type=float, default=0.0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out")
    a = ap.parse_args()
    r = run(a.url, a.seconds, a.procs, a.conc, a.repeat)
    r["label"] = a.label
    print(json.dumps({k: v for k, v in r.items() if k != "server_metrics"}))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
