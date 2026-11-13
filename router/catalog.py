"""Model catalog: which Magic Hour models the router can pick, what each supports, and its credit cost.

Costs come from three places, recorded per model in `cost_source`:
  documented - the "from N credits/image" figure in the Magic Hour API reference (magic-hour SDK 0.92)
  observed   - credits_charged reported by the API on our own renders
  assumed    - placeholder until we have a real charge; the service replaces every
               figure with an EMA of observed `credits_charged` once feedback arrives

Quality is NOT in this file. The router learns it from eval data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TASKS = ("image", "video")
IMAGE_RES = ("640px", "1k", "2k", "4k")
VIDEO_RES = ("480p", "720p", "1080p", "4k")
QUALITY_BARS = ("draft", "standard", "premium")
ASPECTS = ("1:1", "16:9", "9:16")

# Resolution multipliers relative to the reference resolution. Assumed; corrected online.
IMAGE_RES_MULT = {"640px": 1.0, "1k": 1.0, "2k": 2.0, "4k": 4.0}
VIDEO_RES_MULT = {"480p": 0.45, "720p": 0.67, "1080p": 1.0, "4k": 2.5}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str
    base_cost: float  # image: credits/image at 1k. video: credits/second at 1080p.
    cost_source: str
    resolutions: tuple[str, ...]
    max_seconds: int = 0
    notes: str = ""

    def credits(self, resolution: str, seconds: float = 0.0) -> float:
        if self.task == "image":
            return self.base_cost * IMAGE_RES_MULT[resolution]
        return self.base_cost * seconds * VIDEO_RES_MULT[resolution]

    def supports(self, resolution: str, seconds: float = 0.0) -> bool:
        if resolution not in self.resolutions:
            return False
        return self.task == "image" or seconds <= self.max_seconds


IMAGE_MODELS = [
    ModelSpec("z-image-turbo", "image", 5, "documented+observed", ("640px", "1k", "2k"),
              notes="5 credits at 640px observed on 12 real renders"),
    ModelSpec("flux-schnell", "image", 5, "documented", ("640px", "1k", "2k")),
    ModelSpec("qwen-image-2.1", "image", 10, "documented", ("640px", "1k", "2k")),
    ModelSpec("krea-2", "image", 10, "documented", ("640px", "1k")),
    ModelSpec("seedream-v4", "image", 40, "documented+observed", ("640px", "1k", "2k", "4k")),
    ModelSpec("gpt-image-2", "image", 50, "documented", ("640px", "1k", "2k", "4k")),
    ModelSpec("seedream-v5-pro", "image", 75, "documented", ("640px", "1k", "2k")),
    ModelSpec("nano-banana-pro", "image", 150, "documented", ("1k", "2k", "4k")),
]

VIDEO_MODELS = [
    # 180 credits for 5 s @720p observed (image_to_video) -> 36/s @720p -> 54/s @1080p
    ModelSpec("kling-2.6", "video", 54, "observed", ("720p", "1080p"), max_seconds=10),
    # ltx-2: 240 credits for 5 s @720p observed -> 72/s @1080p; ltx-2.5 assumed same
    ModelSpec("ltx-2.5", "video", 72, "observed (ltx-2)", ("480p", "720p", "1080p"), max_seconds=60),
    # 360 credits for 5 s @1080p observed
    ModelSpec("kling-3.0", "video", 72, "observed", ("720p", "1080p", "4k"), max_seconds=15),
    ModelSpec("seedance-2.0-mini", "video", 150, "assumed", ("480p", "720p"), max_seconds=15),
    ModelSpec("veo3.1-lite", "video", 200, "assumed", ("720p", "1080p"), max_seconds=56),
    # 2,640 credits for 5 s @1080p observed
    ModelSpec("seedance-2.0", "video", 528, "observed", ("480p", "720p", "1080p", "4k"), max_seconds=15),
]

ALL_MODELS = IMAGE_MODELS + VIDEO_MODELS
MODEL_INDEX = {m.name: i for i, m in enumerate(ALL_MODELS)}
BY_NAME = {m.name: m for m in ALL_MODELS}
MODELS_FOR_TASK = {t: [m for m in ALL_MODELS if m.task == t] for t in TASKS}

# What a quality-first customer uses today without a router: the top model for the task.
PREMIUM_DEFAULT = {"image": "nano-banana-pro", "video": "seedance-2.0"}


@dataclass
class Request:
    task: str
    prompt: str
    resolution: str
    aspect_ratio: str = "16:9"
    seconds: float = 0.0
    has_image_input: bool = False
    quality: str = "standard"
    allowed_models: list[str] | None = None
    max_credits: float | None = None
    extra: dict = field(default_factory=dict)

    def eligible(self) -> list[ModelSpec]:
        out = []
        for m in MODELS_FOR_TASK[self.task]:
            if self.allowed_models and m.name not in self.allowed_models:
                continue
            if not m.supports(self.resolution, self.seconds):
                continue
            out.append(m)
        return out
