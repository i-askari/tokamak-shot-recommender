"""Recommended trajectory: what to command, based on similar shots that worked.

Takes the nearest neighbors from ShotIndex, keeps the ones that hit their
targets, and blends their *commanded* plasma-current and NBI-power waveforms
(similarity-weighted) onto a common time grid. The spread across those shots
is returned as an envelope so the operator sees how much they varied.
"""

import numpy as np

SIGNALS = {
    # signal key: (scale applied to target data, use abs, display units)
    "ip": (1e-6, True, "MA"),
    "nbi_power": (1.0, False, "MW"),
}


def recommend_trajectory(neighbors, series_by_shot, t_max_ms=6000, dt_ms=50):
    """neighbors: output of ShotIndex.similar_to_shot / match_specs.
    series_by_shot: {shot_number: target/actual JSON (see success.py)}.
    """
    candidates = [r for r in neighbors if r["shot_number"] in series_by_shot]
    used = [r for r in candidates if r.get("overall_success") is True] or candidates
    if not used:
        return None

    grid = np.arange(0, t_max_ms + dt_ms, dt_ms)
    weights = np.array([max(r["similarity"], 1e-6) for r in used])
    out = {"time_ms": grid, "used_shots": [r["shot_number"] for r in used],
           "only_successful": used is not candidates, "signals": {}}

    for key, (scale, use_abs, units) in SIGNALS.items():
        curves, w = [], []
        for r, wi in zip(used, weights):
            target = series_by_shot[r["shot_number"]].get(key, {}).get("target", {})
            if not target.get("time"):
                continue
            y = np.asarray(target["data"], dtype=float) * scale
            if use_abs:
                y = np.abs(y)
            curves.append(np.interp(grid, target["time"], y, left=0.0, right=0.0))
            w.append(wi)
        if not curves:
            continue
        curves = np.vstack(curves)
        out["signals"][key] = {
            "units": units,
            "recommended": np.average(curves, axis=0, weights=w),
            "low": curves.min(axis=0),
            "high": curves.max(axis=0),
        }
    return out
