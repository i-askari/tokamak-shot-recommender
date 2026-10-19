"""Synthetic tokamak shot generator for the public demo.

The real DIII-D archive is restricted, so this module fabricates a database
with the *same schema* as the real pipeline outputs, using simple, plausible
physics scalings. Every shot is generated as raw waveforms + logbook text and
then pushed through the same parametrization, success metric and sentiment
code as the real data, so the demo exercises the real algorithms end to end.

Synthetic shot numbers start at 900000 so they can never be mistaken for real
DIII-D discharges. None of the numbers here describe real experiments.
"""

import numpy as np
import pandas as pd

from .parameters import ECH_TOWERS, GAS_VALVES, NBI_BEAMS, PARAM_COLUMNS
from .sentiment import analyze_sentiment, clean_html, parse_session_html_by_shot
from .success import compute_shot_success

T_TARGET = np.arange(0, 6001, 10.0)   # ms, commanded waveforms
T_ACTUAL = np.arange(0, 6001, 20.0)   # ms, measured waveforms

# name: (weight, ip MA, n_beams, n_gyrotrons, kappa, tritop, greenwald fraction,
#        betaN multiplier, gas-heavy, scan options, goal text)
SCENARIOS = {
    "Ohmic L-mode":         (0.10, (0.6, 1.0), (0, 0), (0, 0), (1.3, 1.7), (0.1, 0.4), (0.3, 0.5), 0.6, False,
                             ["ip", "density"], "Ohmic L-mode reference discharges for diagnostic calibration"),
    "ELMy H-mode":          (0.22, (1.0, 1.4), (2, 4), (0, 2), (1.7, 1.85), (0.4, 0.7), (0.5, 0.75), 1.0, False,
                             ["power", "density", "ip"], "ELMy H-mode confinement scaling"),
    "QH-mode":              (0.10, (1.0, 1.3), (3, 5), (0, 2), (1.75, 1.9), (0.5, 0.7), (0.25, 0.45), 1.0, False,
                             ["power", "none"], "Quiescent H-mode with counter-injection"),
    "Hybrid / adv. inductive": (0.14, (1.0, 1.3), (4, 6), (3, 6), (1.75, 1.9), (0.4, 0.6), (0.5, 0.7), 1.25, False,
                             ["power", "none"], "High-betaN hybrid scenario development"),
    "Steady-state high-βP": (0.10, (0.6, 0.9), (4, 7), (5, 8), (1.8, 2.0), (0.5, 0.7), (0.6, 0.85), 1.35, False,
                             ["power", "ip"], "Fully non-inductive steady-state operation"),
    "Negative triangularity": (0.10, (0.6, 0.9), (1, 3), (0, 3), (1.2, 1.5), (-0.5, -0.3), (0.4, 0.7), 0.9, False,
                             ["power", "density"], "Negative-triangularity L-mode edge without ELMs"),
    "Radiative divertor":   (0.14, (1.0, 1.4), (3, 5), (0, 3), (1.7, 1.85), (0.4, 0.6), (0.7, 0.95), 0.9, True,
                             ["density", "none"], "Divertor detachment with impurity / gas seeding"),
    "ECH electron heating": (0.10, (0.6, 1.0), (0, 1), (4, 8), (1.5, 1.8), (0.2, 0.5), (0.3, 0.5), 0.8, False,
                             ["power", "none"], "Electron-heated plasmas for turbulence studies"),
}


def _trapezoid(t, t_up, t_flat_end, t_down, level):
    """0 -> level over [0, t_up], hold until t_flat_end, back to 0 over t_down."""
    y = np.interp(t, [0, t_up, t_flat_end, t_flat_end + t_down], [0, level, level, 0], right=0.0)
    return y


def _betan(mult, p_heat_mw, ip_ma):
    """Normalized beta rises with heating power and falls with current; capped near the no-wall limit."""
    return min(mult * (0.3 + 0.45 * p_heat_mw ** 0.6 / ip_ma ** 0.5), 4.0)


def _peak_mean(y):
    nz = y[y > 0]
    if len(nz) == 0:
        return None, None
    return float(np.percentile(nz, 95)), float(np.mean(nz))


def _session_config(rng, sid, scenario):
    (w, ip, nb, ng, kap, tri, fgw, bmult, gassy, scans, goal) = SCENARIOS[scenario]
    beams = list(rng.choice(NBI_BEAMS, size=int(rng.integers(nb[0], nb[1] + 1)), replace=False))
    gyros = list(rng.choice(ECH_TOWERS, size=int(rng.integers(ng[0], ng[1] + 1)), replace=False))
    valves = list(rng.choice(GAS_VALVES, size=int(rng.integers(1, 3)), replace=False))
    if gassy:
        valves += [v for v in ("PFX1", "LOB1") if v not in valves]
    return {
        "session": sid, "scenario": scenario, "goal": goal,
        "ip": rng.uniform(*ip), "beams": beams,
        "beam_power": {b: rng.uniform(1.8, 2.5) for b in beams},
        "gyros": gyros, "gyro_power": {g: rng.uniform(0.45, 0.75) for g in gyros},
        "valves": valves, "valve_flow": {v: rng.uniform(15, 40) * (2.5 if gassy else 1) for v in valves},
        "kappa": rng.uniform(*kap), "tritop": rng.uniform(*tri), "fgw": rng.uniform(*fgw),
        "bt": rng.uniform(1.6, 2.1), "betan_mult": bmult,
        "t_up": rng.uniform(600, 1200), "t_flat_end": rng.uniform(3000, 5000), "t_down": rng.uniform(600, 1000),
        "ip_sign": -1.0 if rng.random() < 0.25 else 1.0,   # some target signals are sign-flipped
        "scan": scans[int(rng.integers(len(scans)))],
        "tangential_ts": sid >= 40,                          # later-era diagnostic
    }


def _make_shot(rng, cfg, shot, frac):
    """One discharge: commanded + measured waveforms, scalar parameters, outcome tags."""
    ip = cfg["ip"] * (1 + 0.25 * (frac - 0.5) if cfg["scan"] == "ip" else 1) * rng.normal(1, 0.015)
    pscale = 0.7 + 0.6 * frac if cfg["scan"] == "power" else 1.0
    gscale = 0.6 + 0.9 * frac if cfg["scan"] == "density" else 1.0
    t_up, t_fe, t_dn = cfg["t_up"], cfg["t_flat_end"] * rng.normal(1, 0.02), cfg["t_down"]

    # --- commanded (target) waveforms
    ip_target = cfg["ip_sign"] * _trapezoid(T_TARGET, t_up, t_fe, t_dn, ip * 1e6)
    t_on, t_off = t_up - rng.uniform(100, 400), t_fe
    beam_cmd = {b: cfg["beam_power"][b] * pscale * rng.normal(1, 0.01) for b in cfg["beams"]}
    on_target = ((T_TARGET >= t_on) & (T_TARGET <= t_off)).astype(float)
    nbi_target = on_target * sum(beam_cmd.values())

    # --- what went wrong (if anything)
    p_heat_cmd = sum(beam_cmd.values()) + sum(cfg["gyro_power"].values()) * pscale + 0.6
    betan_est = _betan(cfg["betan_mult"], p_heat_cmd, ip)
    fgw = min(cfg["fgw"] * gscale, 1.1)
    p_disrupt = 0.08 + 0.25 * max(0, betan_est - 3.0) + 0.6 * max(0, fgw - 0.85)
    events = {"disruption": None, "beam_fault": [], "beam_trip": {}, "ip_shortfall": False, "no_plasma": False}
    if rng.random() < 0.02:
        events["no_plasma"] = True
    elif rng.random() < p_disrupt:
        early = rng.random() < 0.5
        events["disruption"] = rng.uniform(0.3 * t_up, t_up) if early else rng.uniform(t_up, t_fe)
    if rng.random() < 0.07:
        events["ip_shortfall"] = True
    for b in cfg["beams"]:
        r = rng.random()
        if r < 0.03:
            events["beam_fault"].append(b)
        elif r < 0.06:
            events["beam_trip"][b] = rng.uniform(t_on + 200, t_off)
    t_end = events["disruption"] if events["disruption"] is not None else np.inf

    # --- measured (actual) waveforms
    level = 0.8 if events["ip_shortfall"] else 1.0
    ip_ideal = _trapezoid(T_ACTUAL, t_up, t_fe, t_dn, ip * 1e6 * level * rng.normal(1, 0.015))
    tau = 30.0
    ip_act = np.zeros_like(ip_ideal)
    for i in range(1, len(T_ACTUAL)):
        ip_act[i] = ip_act[i - 1] + (ip_ideal[i] - ip_act[i - 1]) * (1 - np.exp(-20.0 / tau))
    ip_act *= 1 + rng.normal(0, 0.004, len(ip_act))
    ip_act[T_ACTUAL > t_end] = 0.0
    if events["no_plasma"]:
        ip_act = np.clip(ip_act * 0.05, 0, None)
    ip_act = np.clip(ip_act, 0, None)

    beam_act = {}
    for b in cfg["beams"]:
        on = (T_ACTUAL >= t_on) & (T_ACTUAL <= min(t_off, t_end + 20, events["beam_trip"].get(b, np.inf)))
        if b in events["beam_fault"] or events["no_plasma"]:
            on[:] = False
        beam_act[b] = on * beam_cmd[b] * 1e6 * rng.uniform(0.97, 0.995)
    nbi_act = sum(beam_act.values()) if beam_act else np.zeros_like(T_ACTUAL)

    series = {
        "ip": {"actual": {"time": T_ACTUAL.tolist(), "data": ip_act.tolist()},
               "target": {"time": T_TARGET.tolist(), "data": ip_target.tolist()}},
        "nbi_power": {"actual": {"time": T_ACTUAL.tolist(), "data": np.asarray(nbi_act).tolist(), "units": "W"},
                      "target": {"time": T_TARGET.tolist(), "data": nbi_target.tolist(), "units": "MW"}},
    }

    # --- scalar parameters, reduced exactly like notebook 01/02
    row = {"shot_number": shot, "session": cfg["session"], "scenario": cfg["scenario"]}
    row["peak_ip"], row["mean_ip"] = _peak_mean(ip_act)
    for b in NBI_BEAMS:
        p, m = _peak_mean(beam_act[b]) if b in beam_act else (None, None)
        row[f"peak_pabs_{b}"], row[f"mean_pabs_{b}"] = p, m
    for g in ECH_TOWERS:
        fired = g in cfg["gyros"] and rng.random() > 0.05 and not events["no_plasma"]
        p = cfg["gyro_power"][g] * pscale * 1e6 * rng.normal(1, 0.02) if fired else None
        row[f"peak_echpwr_{g}"], row[f"mean_echpwr_{g}"] = p, (p * rng.uniform(0.85, 0.98) if p else None)
    for v in GAS_VALVES:
        f = cfg["valve_flow"][v] * gscale * rng.normal(1, 0.05) if v in cfg["valves"] else None
        row[f"peak_gasflow_{v}"], row[f"mean_gasflow_{v}"] = f, (f * rng.uniform(0.3, 0.6) if f else None)

    delivered = (sum(row[f"mean_pabs_{b}"] or 0 for b in NBI_BEAMS)
                 + sum(row[f"mean_echpwr_{g}"] or 0 for g in ECH_TOWERS)) / 1e6 + 0.6
    lasted = 1.0 if events["disruption"] is None else max(0.2, (events["disruption"] - t_up) / (t_fe - t_up) + 0.3)
    lasted = 0.1 if events["no_plasma"] else min(lasted, 1.0)
    n19 = 8.8 * ip * fgw * rng.normal(1, 0.04)
    te = (600 + 900 * delivered ** 0.7 / n19 ** 0.5) * rng.normal(1, 0.05)
    row["peak_density"], row["mean_density"] = n19 * 1e19, n19 * 1e19 * rng.uniform(0.4, 0.55) * lasted
    row["peak_temp"], row["mean_temp"] = te, te * rng.uniform(0.35, 0.5) * lasted
    if cfg["tangential_ts"]:
        row["peak_ts_tangential_density"] = row["peak_density"] * rng.uniform(0.85, 0.95)
        row["mean_ts_tangential_density"] = row["mean_density"] * rng.uniform(0.85, 0.95)
        row["peak_ts_tangential_temp"] = te * rng.uniform(0.6, 0.8)
        row["mean_ts_tangential_temp"] = row["mean_temp"] * rng.uniform(0.6, 0.8)
    else:
        for k in ("peak_ts_tangential_density", "mean_ts_tangential_density",
                  "peak_ts_tangential_temp", "mean_ts_tangential_temp"):
            row[k] = None
    row["kappa"] = cfg["kappa"] * rng.normal(1, 0.01)
    row["tritop"] = cfg["tritop"] + rng.normal(0, 0.015)
    row["tribot"] = row["tritop"] * rng.uniform(0.85, 1.1)
    row["betan"] = _betan(cfg["betan_mult"], delivered, ip) * rng.normal(1, 0.05) * (0.6 + 0.4 * lasted)
    row["q95"] = 2.6 * (1 + row["kappa"] ** 2) / 2 / ip * (cfg["bt"] / 2.0) * rng.normal(1, 0.02)
    row["peak_nbi"] = _peak_mean(np.asarray(nbi_act))[0] or 0.0
    return row, series, events


def _notes_for(rng, shot, events, scenario):
    """Templated operator logbook entries consistent with what happened."""
    notes = []
    if events["no_plasma"]:
        notes.append(("CHIEF_OPERATOR", "No breakdown, shot aborted. Problem with the prefill."))
    if events["disruption"] is not None:
        notes.append(("PHYSICS_OPERATOR", f"Disruption at {events['disruption'] / 1000:.2f} s, unfortunately. "
                      + rng.choice(["Locked mode is the likely culprit.", "Density limit, too far.", "Tearing mode grew early."])))
    for b in events["beam_fault"]:
        notes.append(("BEAMS", f"{b} failed to inject - ion source issue."))
    for b, t in events["beam_trip"].items():
        notes.append(("BEAMS", f"{b} trip at {t / 1000:.1f} s."))
    if events["ip_shortfall"]:
        notes.append(("PHYSICS_OPERATOR", "Missed Ip target by ~20%, ohmic flux problem."))
    if not notes:
        notes.append(("PHYSICS_OPERATOR", rng.choice([
            "Good shot, matched previous.", "Nice stable flattop, ran to full length.",
            f"Repeat of last shot, worked well. {scenario} looks good.", "Ok shot, all systems nominal.",
            "Better than previous shot, stable through the flattop."])))
    if rng.random() < 0.15:
        notes.append(("DIAGNOSTICS", rng.choice(["Thomson data ok.", "ECE channel issue, rest of profiles good.",
                                                  "Bad CER fit on this one."])))
    return notes


def generate_dataset(n_sessions=110, seed=7):
    rng = np.random.default_rng(seed)
    names = list(SCENARIOS)
    weights = np.array([SCENARIOS[n][0] for n in names])
    weights /= weights.sum()

    rows, series_by_shot, sessions, sentiment_rows = [], {}, [], []
    shot = 900001
    for sid in range(n_sessions):
        scenario = names[int(rng.choice(len(names), p=weights))]
        cfg = _session_config(rng, sid, scenario)
        n_shots = int(rng.integers(6, 16))
        entries, shots, outcomes = [], [], []
        for k in range(n_shots):
            row, series, events = _make_shot(rng, cfg, shot, k / max(n_shots - 1, 1))
            rows.append(row)
            series_by_shot[shot] = series
            shots.append(shot)
            bad = events["disruption"] is not None or events["no_plasma"] or events["beam_fault"]
            outcomes.append(not bad)
            if rng.random() < 0.75:   # ~25% of shots get no logbook entry of their own
                entries += [f"Shot {shot} {role}: {text}" for role, text in _notes_for(rng, shot, events, scenario)]
            shot += 1
        shot += int(rng.integers(5, 60))   # gap between sessions

        good = np.mean(outcomes)
        summary = ("Productive day, most shots ran well and the scan was completed." if good > 0.8 else
                   "Mixed day. Several shots ok but repeated problems with beams and disruptions." if good > 0.55 else
                   "Difficult session, unfortunately. Many disruptions and beam trips; goal missed.")
        html = (f"<html><body><h2>Synthetic session {sid:03d}: {cfg['goal']}</h2>"
                f"<p>SESSION_LEADER SUMMARY {summary}</p><p>ELECTRONIC LOGBOOK ENTRIES</p>"
                + "".join(f"<p>{e}</p>" for e in entries) + "</body></html>")

        # Same two-tier logic as notebook 04: own notes first, session fallback otherwise.
        text = clean_html(html)
        per_shot = parse_session_html_by_shot(text)
        session_sent = analyze_sentiment(text[text.find("SESSION_LEADER SUMMARY"):text.find("ELECTRONIC LOGBOOK")])
        for s in shots:
            res, source = (analyze_sentiment(per_shot[s]), "per_shot_notes") if s in per_shot else (session_sent, "session_fallback")
            sentiment_rows.append({"session": f"S{sid:03d}", "shot_number": s, "sentiment": res["sentiment"],
                                   "positive_count": res["positive_count"], "negative_count": res["negative_count"],
                                   "source": source})
        sessions.append({"session": sid, "scenario": scenario, "goal": cfg["goal"], "scan": cfg["scan"],
                         "shots": shots, "html": html})

    df_params = pd.DataFrame(rows)
    df_success = pd.DataFrame([r for r in (compute_shot_success(s, series_by_shot[s]) for s in df_params["shot_number"])
                               if r is not None])
    df_sentiment = pd.DataFrame(sentiment_rows)
    assert set(PARAM_COLUMNS) <= set(df_params.columns)
    return {"params": df_params, "series": series_by_shot, "success": df_success,
            "sentiment": df_sentiment, "sessions": sessions}
