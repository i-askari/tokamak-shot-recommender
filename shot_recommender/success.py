"""Objective success metric: did the machine deliver what the operators commanded?

Ported from notebooks/05_shot_success_metric.ipynb. Input is the per-shot
target/actual JSON produced by notebooks/03_target_vs_actual_extraction.ipynb:

    {"ip":        {"actual": {"time": [...], "data": [...]},
                   "target": {"time": [...], "data": [...]}},
     "nbi_power": {"actual": {... "units": "W"},
                   "target": {... "units": "MW"}}}
"""

import numpy as np


def align_actual_to_target(target_time, target_data, actual_time, actual_data, actual_scale=1.0):
    """Interpolates actual onto target's (coarser, step-shaped) timestamps."""
    if not target_time or not actual_time:
        return None, None
    target_time, target_data = np.asarray(target_time), np.asarray(target_data)
    actual_data = np.asarray(actual_data) * actual_scale
    aligned_actual = np.interp(target_time, np.asarray(actual_time), actual_data)
    return target_data, aligned_actual


def score_target_vs_actual(series, actual_scale, use_abs, tolerance_pct):
    """Peak-based target-vs-actual score for one signal.

    use_abs takes abs() before comparing, which handles sign-flipped target
    signals (like IPTIPP) without assuming a direction.
    """
    t_time = series.get("target", {}).get("time")
    t_data = series.get("target", {}).get("data")
    a_time = series.get("actual", {}).get("time")
    a_data = series.get("actual", {}).get("data")
    if not (t_time and t_data and a_time and a_data):
        return None

    target_arr, actual_arr = align_actual_to_target(t_time, t_data, a_time, a_data, actual_scale)
    if use_abs:
        target_arr, actual_arr = np.abs(target_arr), np.abs(actual_arr)
    if len(target_arr) == 0:
        return None

    peak_target = float(np.max(target_arr))
    peak_actual = float(np.max(actual_arr))
    if not peak_target:
        return None

    pct_diff = abs(peak_actual - peak_target) / peak_target * 100
    return {
        "target_peak": peak_target,
        "actual_peak": peak_actual,
        "percent_diff": round(pct_diff, 1),
        "hit_target": pct_diff <= tolerance_pct,
    }


def compute_shot_success(shot_number, series, tolerance_pct=15):
    """Scores plasma current and total NBI power independently, then combines.

    A shot is successful only if every signal that could be checked hit its
    target. Returns None when neither signal is usable.
    """
    ip = score_target_vs_actual(series.get("ip", {}), actual_scale=1.0,
                                use_abs=True, tolerance_pct=tolerance_pct)
    nbi = score_target_vs_actual(series.get("nbi_power", {}), actual_scale=1e-6,
                                 use_abs=False, tolerance_pct=tolerance_pct)
    if ip is None and nbi is None:
        return None

    row = {
        "shot_number": shot_number,
        "ip_target_peak_A": ip["target_peak"] if ip else None,
        "ip_actual_peak_A": ip["actual_peak"] if ip else None,
        "ip_percent_diff": ip["percent_diff"] if ip else None,
        "ip_hit_target": ip["hit_target"] if ip else None,
        "nbi_target_peak_MW": nbi["target_peak"] if nbi else None,
        "nbi_actual_peak_MW": nbi["actual_peak"] if nbi else None,
        "nbi_percent_diff": nbi["percent_diff"] if nbi else None,
        "nbi_hit_target": nbi["hit_target"] if nbi else None,
    }
    known_hits = [v for v in (row["ip_hit_target"], row["nbi_hit_target"]) if v is not None]
    row["overall_success"] = all(known_hits)
    return row
