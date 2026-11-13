"""Render the README charts from results/*.json."""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S1, S2, S3, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c87"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def frontier(rep, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160, facecolor=SURF)
    style(ax)
    series = [("cascade[passnet]", "Router + judge, escalate on reject", S2, "-"),
              ("passnet", "Router (single render)", S1, "-"),
              ("bucket_rules", "Hand-built rules per bucket", S3, "-"),
              ("oracle", "Oracle (knows hidden model skills)", MUTED, "--")]
    for key, label, col, ls in series:
        pts = sorted(rep["pareto"][key])
        ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts], color=col, lw=2, ls=ls, label=label)
    px, py = rep["premium_point"]
    ax.scatter([px], [py * 100], s=60, color=INK, zorder=5)
    ax.annotate("Always premium model", (px, py * 100), xytext=(-10, -22), textcoords="offset points",
                ha="right", color=INK, fontsize=9)
    ax.axhline(py * 100, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Credits per request", color=INK2, fontsize=9)
    ax.set_ylabel("Renders that pass the quality bar (%)", color=INK2, fontsize=9)
    ax.set_ylim(70, 100)
    ax.set_title("Cost vs quality on 36k held-out requests", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)


def drift(runs, out):
    labels = {"static": ("No online learning", MUTED), "bias": ("Online bias correction", S1),
              "bias_explore_retrain": ("Bias correction + 5% exploration", S2)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160, facecolor=SURF)
    for ax in axes:
        style(ax)
    for r in runs:
        lab, col = labels.get(r["arm"], (r["arm"], INK))
        xs = [w["start"] / 1000 for w in r["windows"]]
        axes[0].plot(xs, [(w["pass"] - w["prem_pass"]) * 100 for w in r["windows"]], color=col, lw=2, marker="o", ms=4, label=lab)
        axes[1].plot(xs, [(1 - w["credits_per_req"] / w["prem_credits_per_req"]) * 100 for w in r["windows"]],
                     color=col, lw=2, marker="o", ms=4, label=lab)
    d = runs[0]["drift_at"] / 1000
    for ax in axes:
        ax.axvline(d, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("Requests served (thousands)", color=INK2, fontsize=9)
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[0].text(d + 0.2, axes[0].get_ylim()[0] + 0.3, "models change", color=INK2, fontsize=8)
    axes[0].set_title("Pass rate vs always-premium (points)", color=INK, fontsize=10, loc="left")
    axes[1].set_title("Credits saved vs always-premium (%)", color=INK, fontsize=10, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, labelcolor=INK, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, facecolor=SURF)


def main():
    res = pathlib.Path("results")
    if (res / "offline.json").exists():
        frontier(json.loads((res / "offline.json").read_text()), res / "frontier.png")
    if (res / "drift.jsonl").exists():
        runs = [json.loads(l) for l in (res / "drift.jsonl").read_text().splitlines() if l.strip()]
        drift(runs, res / "drift.png")


if __name__ == "__main__":
    main()
