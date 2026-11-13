"""Integration tests: need Postgres + Redis (ROUTER_TEST_PG / ROUTER_TEST_REDIS) and models/passnet.pt."""
import os
import uuid

import pytest

PG = os.environ.get("ROUTER_TEST_PG", "postgresql://postgres:postgres@localhost/router_test")
RD = os.environ.get("ROUTER_TEST_REDIS", "redis://localhost:6379/15")


@pytest.fixture(scope="module")
def client():
    psycopg = pytest.importorskip("psycopg")
    redis = pytest.importorskip("redis")
    try:
        with psycopg.connect(PG.rsplit("/", 1)[0] + "/postgres", autocommit=True) as c:
            c.execute("DROP DATABASE IF EXISTS router_test")
            c.execute("CREATE DATABASE router_test")
        redis.from_url(RD).flushdb()
    except Exception as e:
        pytest.skip(f"postgres/redis not available: {e}")
    if not os.path.exists("models/passnet.pt"):
        pytest.skip("no trained model")
    os.environ.update(ROUTER_PG=PG, ROUTER_REDIS=RD, ROUTER_NS=f"t{uuid.uuid4().hex[:6]}")
    import importlib
    import router.service as svc
    importlib.reload(svc)
    from fastapi.testclient import TestClient
    with TestClient(svc.app) as c:
        yield c


BODY = {"task": "image", "prompt": "a bowl of fruit, watercolor illustration", "resolution": "1k", "quality": "standard"}


def test_route_returns_eligible_model(client):
    d = client.post("/v1/route", json=BODY).json()
    names = [a["model"] for a in d["alternatives"]]
    assert d["model"] in names and 0 <= d["p_pass"] <= 1
    assert "krea-2" in names and "nano-banana-pro" in names


def test_score_cache_hit_on_repeat(client):
    body = dict(BODY, prompt="a cabin in the woods, pixel art")
    assert client.post("/v1/route", json=body).json()["cache_hit"] is False
    assert client.post("/v1/route", json=body).json()["cache_hit"] is True


def test_unsupported_request_is_422(client):
    r = client.post("/v1/route", json={"task": "video", "prompt": "x", "resolution": "4k", "seconds": 60})
    assert r.status_code == 422


def test_max_credits_is_respected(client):
    d = client.post("/v1/route", json=dict(BODY, quality="premium", max_credits=10)).json()
    assert d["est_credits"] <= 10


def test_feedback_is_idempotent_and_moves_bias(client):
    d = client.post("/v1/route", json=dict(BODY, prompt="a red vintage convertible")).json()
    r1 = client.post("/v1/feedback", json={"decision_id": d["decision_id"], "passed": False}).json()
    r2 = client.post("/v1/feedback", json={"decision_id": d["decision_id"], "passed": False}).json()
    assert r1["applied"] is True and r2["applied"] is False
    m = {x["model"]: x for x in client.get("/v1/models").json()["models"]}
    assert m[d["model"]]["feedback_n"] == 1


def test_unknown_feedback_404(client):
    assert client.post("/v1/feedback", json={"decision_id": str(uuid.uuid4()), "passed": True}).status_code == 404


def test_shadow_feedback_is_recorded_separately(client):
    d = client.post("/v1/route", json=dict(BODY, prompt="a glass chess set, macro shot", quality="premium")).json()
    cheaper = [a["model"] for a in d["alternatives"] if a["est_credits"] < d["est_credits"]]
    assert cheaper, "premium-bar request should have cheaper alternatives"
    a = client.post("/v1/feedback", json={"decision_id": d["decision_id"], "passed": True}).json()
    b = client.post("/v1/feedback", json={"decision_id": d["decision_id"], "passed": False,
                                          "model": cheaper[0], "shadow": True}).json()
    assert a["applied"] and b["applied"]


def test_feedback_for_ineligible_model_is_rejected(client):
    d = client.post("/v1/route", json=dict(BODY, resolution="640px")).json()
    r = client.post("/v1/feedback", json={"decision_id": d["decision_id"], "passed": True, "model": "nano-banana-pro"})
    assert r.status_code == 422
