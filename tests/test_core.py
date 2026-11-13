import numpy as np
import torch

from router.catalog import BY_NAME, Request
from router.features import dense, tokens
from router.model import PassNet, collate, load_bytes, save_bytes
from router.policy import choose


def test_tokens_are_deterministic_and_quote_invariant():
    a = tokens('a neon sign that reads "OPEN LATE"')
    b = tokens('a neon sign that reads "SALE NOW"')
    assert a == tokens('a neon sign that reads "OPEN LATE"')
    assert a == b  # quoted text is reduced to a length marker
    assert a != tokens('a neon sign that reads "SALE"')


def test_dense_features_shape():
    v = dense({"task": "video", "prompt": "x", "resolution": "720p", "seconds": 10})
    assert v.shape[0] > 10 and v.dtype == np.float32


def test_ordinal_head_is_monotone_across_bars():
    torch.manual_seed(0)
    net = PassNet(buckets=1024).eval()
    reqs = [{"task": "image", "prompt": f"prompt {i}", "resolution": "1k"} for i in range(16)]
    tok, off, dn = collate(reqs)
    tok = tok % 1023 + 1
    p = net(tok, off, dn)
    assert torch.all(p[:, :, 0] >= p[:, :, 1]) and torch.all(p[:, :, 1] >= p[:, :, 2])


def test_save_load_roundtrip():
    net = PassNet(buckets=1024)
    net2, meta = load_bytes(save_bytes(net, {"version": "t", "arch": {"buckets": 1024}}))
    assert meta["version"] == "t"
    for a, b in zip(net.state_dict().values(), net2.state_dict().values()):
        assert torch.equal(a, b)


def _c(name, p, cost):
    return (BY_NAME[name], p, cost)


def test_policy_picks_cheapest_confident():
    c = choose([_c("z-image-turbo", 0.95, 5), _c("seedream-v4", 0.97, 40), _c("nano-banana-pro", 0.99, 150)], 0.9)
    assert c.model == "z-image-turbo" and c.reason == "cheapest_confident"


def test_policy_falls_back_to_best():
    c = choose([_c("z-image-turbo", 0.3, 5), _c("nano-banana-pro", 0.8, 150)], 0.9)
    assert c.model == "nano-banana-pro" and c.reason == "fallback_best"


def test_policy_respects_max_credits():
    c = choose([_c("z-image-turbo", 0.3, 5), _c("nano-banana-pro", 0.99, 150)], 0.9, max_credits=50)
    assert c.model == "z-image-turbo"


def test_eligibility_filters_resolution_and_duration():
    r = Request(task="video", prompt="x", resolution="4k", seconds=30)
    assert r.eligible() == []
    r = Request(task="image", prompt="x", resolution="640px")
    assert "nano-banana-pro" not in [m.name for m in r.eligible()]
