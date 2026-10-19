"""Cosine-similarity search over the normalized 77-parameter shot vectors.

Ported from notebooks/07_similarity_search.ipynb into a small reusable class.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from .parameters import PARAM_COLUMNS


class ShotIndex:
    """Holds the normalized parameter matrix plus the per-shot enrichment tables.

    df_params:    one row per shot, `shot_number` + PARAM_COLUMNS
    df_success:   output of the success metric (optional)
    df_sentiment: output of the logbook sentiment pass (optional)
    """

    def __init__(self, df_params, df_success=None, df_sentiment=None, param_columns=PARAM_COLUMNS):
        self.param_columns = list(param_columns)
        self.df_params = df_params[["shot_number"] + self.param_columns].copy()
        # A diagnostic that never fired (beam off, gyrotron off) is a real zero.
        self.df_params[self.param_columns] = self.df_params[self.param_columns].astype(float).fillna(0)

        self.scaler = MinMaxScaler()
        self.vectors = self.scaler.fit_transform(self.df_params[self.param_columns])
        self.shot_numbers = self.df_params["shot_number"].to_numpy()
        self._row_of = {int(s): i for i, s in enumerate(self.shot_numbers)}

        self._success = {}
        if df_success is not None:
            self._success = {int(r["shot_number"]): r for r in df_success.to_dict("records")}

        # Prefer a shot's own logbook notes over the session-wide fallback.
        self._sentiment = {}
        if df_sentiment is not None:
            for r in df_sentiment.to_dict("records"):
                shot = int(r["shot_number"])
                existing = self._sentiment.get(shot)
                if existing is None or (existing["source"] == "session_fallback"
                                        and r["source"] == "per_shot_notes"):
                    self._sentiment[shot] = r

    def __len__(self):
        return len(self.shot_numbers)

    def enrich(self, shot_number):
        s = self._sentiment.get(shot_number, {})
        ok = self._success.get(shot_number, {})
        return {
            "sentiment": s.get("sentiment"),
            "sentiment_source": s.get("source"),
            "overall_success": ok.get("overall_success"),
            "ip_percent_diff": ok.get("ip_percent_diff"),
            "nbi_percent_diff": ok.get("nbi_percent_diff"),
        }

    def _rank(self, query_vector, n, exclude=None):
        similarities = cosine_similarity(query_vector.reshape(1, -1), self.vectors)[0]
        results = []
        for idx in np.argsort(similarities)[::-1]:
            shot = int(self.shot_numbers[idx])
            if shot == exclude:
                continue
            entry = {"shot_number": shot, "similarity": round(float(similarities[idx]), 4)}
            entry.update(self.enrich(shot))
            results.append(entry)
            if len(results) == n:
                break
        return results

    def similar_to_shot(self, shot_number, n=5):
        """Nearest neighbors of a shot already in the database."""
        if shot_number not in self._row_of:
            raise KeyError(f"Shot {shot_number} not found in database")
        return self._rank(self.vectors[self._row_of[shot_number]], n, exclude=shot_number)

    def match_specs(self, desired_specs, n=5):
        """Closest historical shots to a planned setup.

        Unspecified parameters default to the database mean, as in the notebook.
        """
        unknown = set(desired_specs) - set(self.param_columns)
        if unknown:
            raise KeyError(f"Unknown parameter(s): {sorted(unknown)}")
        query = self.df_params[self.param_columns].mean().to_dict()
        query.update(desired_specs)
        query_vector = self.scaler.transform(pd.DataFrame([query], columns=self.param_columns))[0]
        return self._rank(query_vector, n)
