"""Model registry CLI.

    python -m router.registry register models/passnet.pt --tau models/tau.json [--metrics results/offline.json] [--activate]
    python -m router.registry activate <version>
    python -m router.registry list

Activation is one transaction (the partial unique index allows a single active row), then a Redis
PUBLISH makes every API worker hot-swap the model without a restart. The online bias table is reset,
because a retrained model has already absorbed the feedback the biases were correcting for.
"""
from __future__ import annotations

import argparse
import json
import os

import psycopg
import redis

from router.model import load_bytes

PG = os.environ.get("ROUTER_PG", "postgresql://postgres:postgres@localhost/router")
RD = os.environ.get("ROUTER_REDIS", "redis://localhost:6379/0")
NS = os.environ.get("ROUTER_NS", "rt")


def register(path: str, tau_path: str, metrics_path: str | None, activate_now: bool) -> str:
    blob = open(path, "rb").read()
    _, meta = load_bytes(blob)
    tau = json.load(open(tau_path))
    metrics = json.load(open(metrics_path)) if metrics_path else {}
    metrics = {k: metrics[k] for k in ("prediction", "routing") if k in metrics}
    with psycopg.connect(PG) as c:
        c.execute(open(os.path.join(os.path.dirname(__file__), "schema.sql")).read())
        c.execute("INSERT INTO model_registry (version, artifact, tau, metrics) VALUES (%s,%s,%s,%s) "
                  "ON CONFLICT (version) DO UPDATE SET tau = EXCLUDED.tau, metrics = EXCLUDED.metrics",
                  (meta["version"], blob, json.dumps(tau), json.dumps(metrics)))
    if activate_now:
        activate(meta["version"])
    return meta["version"]


def activate(version: str):
    with psycopg.connect(PG) as c:
        with c.transaction():
            if c.execute("SELECT 1 FROM model_registry WHERE version = %s", (version,)).fetchone() is None:
                raise SystemExit(f"unknown version {version}")
            c.execute("UPDATE model_registry SET active = false WHERE active")
            c.execute("UPDATE model_registry SET active = true WHERE version = %s", (version,))
    r = redis.from_url(RD)
    r.delete(f"{NS}:bias")
    r.publish(f"{NS}:model", version)
    print(f"activated {version}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("register")
    a.add_argument("path")
    a.add_argument("--tau", required=True)
    a.add_argument("--metrics")
    a.add_argument("--activate", action="store_true")
    b = sub.add_parser("activate")
    b.add_argument("version")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "register":
        print(register(args.path, args.tau, args.metrics, args.activate))
    elif args.cmd == "activate":
        activate(args.version)
    else:
        with psycopg.connect(PG) as c:
            for row in c.execute("SELECT version, active, created_at FROM model_registry ORDER BY created_at"):
                print(*row)


if __name__ == "__main__":
    main()
