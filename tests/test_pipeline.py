import numpy as np
import pandas as pd
import pytest

from shot_recommender import PARAM_COLUMNS, ShotIndex, analyze_sentiment, compute_shot_success, recommend_trajectory
from shot_recommender.sentiment import clean_html, parse_session_html_by_shot
from shot_recommender.synthetic import generate_dataset


@pytest.fixture(scope="module")
def data():
    return generate_dataset(n_sessions=30, seed=1)


def _series(target_peak, actual_peak, sign=1.0):
    t = list(np.arange(0, 6001, 10.0))
    trap = np.interp(t, [0, 1000, 4000, 5000], [0, 1, 1, 0])
    return {"target": {"time": t, "data": list(sign * target_peak * trap)},
            "actual": {"time": t, "data": list(actual_peak * trap)}}


def test_feature_vector_is_77_dimensional():
    assert len(PARAM_COLUMNS) == 77
    assert len(set(PARAM_COLUMNS)) == 77


def test_success_hits_within_tolerance():
    # The worked example from the real data: 1.20 MA target, 1.16 MA actual -> 3.2 %
    row = compute_shot_success(1, {"ip": _series(1.2e6, 1.162109e6),
                                   "nbi_power": _series(2.236, 2.1988e6)})
    assert row["ip_percent_diff"] == pytest.approx(3.2, abs=0.1)
    assert row["overall_success"] is True


def test_success_misses_and_handles_sign_flipped_target():
    row = compute_shot_success(1, {"ip": _series(1.0e6, 0.8e6, sign=-1.0)})
    assert row["ip_percent_diff"] == pytest.approx(20.0, abs=0.1)
    assert row["ip_hit_target"] is False
    assert row["nbi_hit_target"] is None
    assert row["overall_success"] is False


def test_success_none_without_usable_signals():
    assert compute_shot_success(1, {}) is None


def test_sentiment_ignores_plasma_vocabulary():
    # "mode" and "clean" must not count as positive words.
    assert analyze_sentiment("H-mode after wall cleaning")["sentiment"] == "mixed"
    assert analyze_sentiment("Disruption at 2.1 s, unfortunately")["sentiment"] == "negative"
    assert analyze_sentiment("Good shot, matched previous")["sentiment"] == "positive"


def test_logbook_entries_are_attributed_to_their_own_shot():
    html = ("<p>ELECTRONIC LOGBOOK ENTRIES</p><p>Shot 900001 PHYSICS_OPERATOR: Good shot.</p>"
            "<p>Shot 900002 BEAMS: 30L trip.</p><p>Shot 900002 VISITOR: ignored role.</p>")
    notes = parse_session_html_by_shot(clean_html(html))
    assert set(notes) == {900001, 900002}
    assert "trip" in notes[900002] and "ignored" not in notes[900002]


def test_similar_to_shot_excludes_query_and_ranks_session_mates(data):
    idx = ShotIndex(data["params"], data["success"], data["sentiment"])
    session_of = data["params"].set_index("shot_number")["session"]
    query = int(data["params"]["shot_number"].iloc[50])
    results = idx.similar_to_shot(query, n=5)
    assert len(results) == 5
    assert query not in [r["shot_number"] for r in results]
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)
    # Shots from the same session share a configuration, so they should dominate.
    assert session_of[results[0]["shot_number"]] == session_of[query]


def test_match_specs_rejects_unknown_parameters(data):
    idx = ShotIndex(data["params"])
    with pytest.raises(KeyError):
        idx.match_specs({"warp_factor": 9})
    assert len(idx.match_specs({"peak_ip": 1.0e6, "kappa": 1.8}, n=3)) == 3


def test_sentiment_prefers_per_shot_notes_over_fallback():
    params = pd.DataFrame([{"shot_number": 1, **{c: 1.0 for c in PARAM_COLUMNS}},
                           {"shot_number": 2, **{c: 0.5 for c in PARAM_COLUMNS}}])
    sentiment = pd.DataFrame([
        {"shot_number": 1, "sentiment": "negative", "source": "session_fallback"},
        {"shot_number": 1, "sentiment": "positive", "source": "per_shot_notes"},
    ])
    idx = ShotIndex(params, df_sentiment=sentiment)
    assert idx.enrich(1)["sentiment"] == "positive"


def test_trajectory_uses_only_successful_neighbors(data):
    idx = ShotIndex(data["params"], data["success"], data["sentiment"])
    neighbors = idx.similar_to_shot(int(data["params"]["shot_number"].iloc[10]), n=5)
    traj = recommend_trajectory(neighbors, data["series"])
    ok = {r["shot_number"] for r in neighbors if r["overall_success"] is True}
    if ok:
        assert set(traj["used_shots"]) == ok
    ip = traj["signals"]["ip"]
    assert np.all(ip["low"] <= ip["recommended"] + 1e-9)
    assert np.all(ip["recommended"] <= ip["high"] + 1e-9)
