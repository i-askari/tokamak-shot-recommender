"""Builds the synthetic database, README figures, and the web demo's data file.

    python scripts/build_assets.py

Outputs:
    data/synthetic/*.csv          synthetic equivalents of the real pipeline CSVs
    docs/img/*.png                README / site figures
    docs/data/demo.json           compact database for the in-browser demo
"""

import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore", message=".*encountered in matmul")

from shot_recommender import PARAM_COLUMNS, ShotIndex, recommend_trajectory  # noqa: E402
from shot_recommender.synthetic import SCENARIOS, generate_dataset  # noqa: E402

# Chart tokens (light surface)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})


def save(fig, name):
    path = os.path.join(ROOT, "docs", "img", name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def fig_success(d):
    """One shot that hit its targets and one that missed, target vs. actual."""
    s = d["success"].set_index("shot_number")
    hit = int(s[(s.overall_success) & (s.ip_percent_diff < 2) & (s.nbi_percent_diff < 3)].index[3])
    miss = int(s[(~s.overall_success.astype(bool)) & (s.ip_percent_diff > 25)].index[0])
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True)
    for col, shot in enumerate([hit, miss]):
        ser, r = d["series"][shot], s.loc[shot]
        for row, (key, scale, unit, label) in enumerate([("ip", 1e-6, "MA", "Plasma current"),
                                                          ("nbi_power", 1.0, "MW", "NBI power")]):
            ax = axes[row, col]
            tt, td = np.array(ser[key]["target"]["time"]), np.abs(np.array(ser[key]["target"]["data"])) * scale
            ta, da = np.array(ser[key]["actual"]["time"]), np.array(ser[key]["actual"]["data"]) * (scale if key == "ip" else 1e-6)
            ax.plot(tt / 1000, td, color=MUTED, lw=2, label="Commanded (target)")
            ax.plot(ta / 1000, da, color=BLUE, lw=2, label="Measured (actual)")
            diff = r["ip_percent_diff"] if key == "ip" else r["nbi_percent_diff"]
            ax.set_ylabel(f"{label} ({unit})")
            if np.isfinite(diff):
                ax.text(0.98, 0.9, f"peak diff {diff:.1f}%", transform=ax.transAxes, ha="right",
                        color=INK2, fontsize=9)
            if row == 0:
                ok = bool(r["overall_success"])
                ax.set_title(f"{'✓ Hit target' if ok else '✗ Missed target'}  ·  synthetic shot {shot}", pad=10)
            if row == 1:
                ax.set_xlabel("Time (s)")
    axes[1, 0].legend(loc="center right", fontsize=9)
    fig.text(0.5, -0.02, "Success = peak actual within 15 % of peak target, for both plasma current and NBI power",
             ha="center", color=INK2, fontsize=9)
    fig.tight_layout()
    save(fig, "success_metric.png")


def fig_similarity_map(d, idx, query):
    """2-D PCA projection of the 77-D normalized vectors, with a query and its top-5."""
    coords = PCA(n_components=2, random_state=0).fit_transform(idx.vectors)
    pos = {int(s): i for i, s in enumerate(idx.shot_numbers)}
    top = [r["shot_number"] for r in idx.similar_to_shot(query, n=5)]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(coords[:, 0], coords[:, 1], s=10, color=AXIS, linewidths=0, label="All shots in database")
    q = coords[pos[query]]
    for s in top:
        c = coords[pos[s]]
        ax.plot([q[0], c[0]], [q[1], c[1]], color=BLUE, lw=1, alpha=0.6, zorder=2)
    nb = coords[[pos[s] for s in top]]
    ax.scatter(nb[:, 0], nb[:, 1], s=60, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3,
               label="Top-5 most similar")
    ax.scatter([q[0]], [q[1]], s=140, color=ORANGE, marker="*", edgecolors=SURFACE, linewidths=1.5, zorder=4,
               label=f"Query shot {query}")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Every shot is a point in 77-D parameter space (shown here projected to 2-D)", pad=10)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save(fig, "similarity_map.png")


def fig_trajectory(d, idx, query):
    neighbors = idx.similar_to_shot(query, n=5)
    traj = recommend_trajectory(neighbors, d["series"])
    t = traj["time_ms"] / 1000
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (key, label) in zip(axes, [("ip", "Plasma current"), ("nbi_power", "NBI power")]):
        sig = traj["signals"][key]
        ax.fill_between(t, sig["low"], sig["high"], color=BLUE, alpha=0.12, linewidth=0,
                        label="Range across successful neighbors")
        ax.plot(t, sig["recommended"], color=BLUE, lw=2, label="Recommended command")
        ax.set_title(f"{label} ({sig['units']})", pad=8)
        ax.set_xlabel("Time (s)")
    axes[0].legend(loc="center right", fontsize=9)
    fig.suptitle(f"Recommended trajectory for a shot like {query}, blended from "
                 f"{len(traj['used_shots'])} similar shots that hit their targets",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, "recommended_trajectory.png")


def fig_real_results():
    """Aggregate numbers from the run on the real archive (no per-shot data)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    labels = ["Physics parameters", "Objective success check", "Logbook sentiment", "Experiment goal text",
              "All three"]
    vals = [3718, 3331, 1231, 639, 483]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.55, color=BLUE)
    for yi, v in zip(y, vals):
        ax.text(v + 60, yi, f"{v:,}  ({100 * v / 3718:.0f}%)", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 4700)
    ax.grid(axis="y", visible=False)
    ax.set_title("Coverage across the 3,718 real shots", pad=10)
    ax.set_xlabel("Shots")

    ax = axes[1]
    beams = ["15L", "15R", "21L", "21R", "30L", "30R", "33L", "33R"]
    cov = [60.4, 52.5, 30.8, 36.6, 77.9, 43.0, 66.0, 65.6]
    x = np.arange(len(beams))
    ax.bar(x, cov, width=0.55, color=BLUE)
    for xi, v in zip(x, cov):
        ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(x, beams)
    ax.set_ylim(0, 90)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Shots with beam firing (%)")
    ax.set_title("How often each of the 8 neutral beams fired", pad=10)
    fig.tight_layout()
    save(fig, "real_data_results.png")


def write_csvs(d):
    out = os.path.join(ROOT, "data", "synthetic")
    os.makedirs(out, exist_ok=True)
    d["params"].to_csv(os.path.join(out, "shot_parameters.csv"), index=False, float_format="%.6g")
    d["success"].to_csv(os.path.join(out, "shot_success_scores.csv"), index=False, float_format="%.6g")
    d["sentiment"].to_csv(os.path.join(out, "shot_sentiment.csv"), index=False)
    print("wrote data/synthetic/*.csv")


def write_demo_json(d, idx):
    """Compact file for the browser: normalized vectors as 0..1000 ints, waveforms at 100 ms."""
    params = d["params"].set_index("shot_number")
    success = d["success"].set_index("shot_number")
    sent = {int(s): idx.enrich(int(s)) for s in idx.shot_numbers}
    grid = np.arange(0, 6001, 100.0)
    scen = list(SCENARIOS)
    shots, waves = [], {}
    for i, s in enumerate(idx.shot_numbers):
        s = int(s)
        e = sent[s]
        ok = success.loc[s] if s in success.index else None
        shots.append({
            "n": s, "ses": int(params.loc[s, "session"]), "sc": scen.index(params.loc[s, "scenario"]),
            "v": [int(round(x * 1000)) for x in idx.vectors[i]],
            "ok": None if ok is None else bool(ok["overall_success"]),
            "ipd": None if ok is None or not np.isfinite(ok["ip_percent_diff"]) else float(ok["ip_percent_diff"]),
            "nbd": None if ok is None or not np.isfinite(ok["nbi_percent_diff"]) else float(ok["nbi_percent_diff"]),
            "se": (e["sentiment"] or "")[:1], "src": "o" if e["sentiment_source"] == "per_shot_notes" else "f",
        })
        ser = d["series"][s]
        w = []
        for key, scale in (("ip", 1e-3), ("nbi_power", None)):
            for kind in ("target", "actual"):
                tt, dd = np.array(ser[key][kind]["time"]), np.abs(np.array(ser[key][kind]["data"]))
                k = scale if scale else (1e3 if kind == "target" else 1e-3)   # -> kA, kW
                w.append([int(round(v)) for v in np.interp(grid, tt, dd * k, right=0)])
        waves[s] = w
    sessions = {str(x["session"]): {"goal": x["goal"], "scan": x["scan"]} for x in d["sessions"]}
    out = {
        "columns": PARAM_COLUMNS,
        "min": [float(v) for v in idx.scaler.data_min_],
        "max": [float(v) for v in idx.scaler.data_max_],
        "mean": [float(v) for v in idx.df_params[PARAM_COLUMNS].mean()],
        "scenarios": scen, "sessions": sessions, "t": grid.tolist(),
        "shots": shots, "waves": waves,
    }
    path = os.path.join(ROOT, "docs", "data", "demo.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote docs/data/demo.json", f"{os.path.getsize(path) / 1e6:.2f} MB")


def main():
    d = generate_dataset()
    idx = ShotIndex(d["params"], d["success"], d["sentiment"])
    query = int(d["params"].query("scenario == 'ELMy H-mode'")["shot_number"].iloc[4])
    write_csvs(d)
    fig_success(d)
    fig_similarity_map(d, idx, query)
    fig_trajectory(d, idx, query)
    fig_real_results()
    write_demo_json(d, idx)


if __name__ == "__main__":
    main()
