"""Request -> model input. Pure Python + numpy so the API and the trainer share one code path."""
from __future__ import annotations

import re
import zlib

import numpy as np

from router.catalog import ASPECTS, IMAGE_RES, TASKS, VIDEO_RES

HASH_BUCKETS = 1 << 18
MAX_TOKENS = 96
_WORD = re.compile(r"[a-z0-9']+|[%:$]")
_QUOTED = re.compile(r"\"([^\"]+)\"|'([^']{2,})'")

RES_ALL = IMAGE_RES + VIDEO_RES
DENSE_DIM = len(TASKS) + len(RES_ALL) + len(ASPECTS) + 7


def _h(s: str) -> int:
    return zlib.crc32(s.encode()) % (HASH_BUCKETS - 1) + 1  # 0 is padding


def tokens(prompt: str) -> list[int]:
    """Hashed word unigrams + bigrams. Quoted text is reduced to a length marker so the model
    learns 'renders N words of text' rather than memorising specific slogans."""
    p = prompt.lower()
    p = _QUOTED.sub(lambda m: f" qtext{min(8, len((m.group(1) or m.group(2)).split()))} ", p)
    words = _WORD.findall(p)
    ids = [_h("u:" + w) for w in words]
    ids += [_h("b:" + a + "_" + b) for a, b in zip(words, words[1:])]
    return ids[:MAX_TOKENS] or [_h("u:<empty>")]


def dense(req: dict) -> np.ndarray:
    v = np.zeros(DENSE_DIM, dtype=np.float32)
    o = 0
    v[o + TASKS.index(req["task"])] = 1; o += len(TASKS)
    v[o + RES_ALL.index(req["resolution"])] = 1; o += len(RES_ALL)
    v[o + ASPECTS.index(req.get("aspect_ratio", "16:9"))] = 1; o += len(ASPECTS)
    prompt = req["prompt"]
    quoted = [m.group(1) or m.group(2) for m in _QUOTED.finditer(prompt)]
    qwords = sum(len(q.split()) for q in quoted)
    sec = float(req.get("seconds") or 0)
    v[o:o + 7] = [
        sec / 30.0,
        max(0.0, sec - 5) / 25.0,
        float(bool(req.get("has_image_input"))),
        min(qwords, 12) / 12.0,
        float(qwords > 0),
        min(len(prompt.split()), 80) / 80.0,
        float(bool(re.search(r"\d", prompt))),
    ]
    return v
