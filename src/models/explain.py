"""
SHAP explanation wrapper.

Given a single feature vector, return the top-K SHAP contributions with
human-readable text suitable for display in the web app.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# Templates for translating (feature_name, shap_value) -> English.
HUMANIZE = {
    "opp_def_rating_z": lambda v: (
        f"Opponent defense is very strong (top of the league)" if v > 0
        else f"Opponent defense is below average — easier matchup"
    ),
    "opp_pace_z": lambda v: (
        f"Opponent prefers a slower pace, fewer possessions" if v > 0
        else f"Opponent runs a faster pace, more possessions"
    ),
    "opp_opp_efg_pct_z": lambda v: (
        f"Opponent forces poor shooting" if v < 0
        else f"Opponent allows efficient scoring"
    ),
    "rs_game_score_std_z": lambda v: (
        f"Player has been streaky in regular season" if v > 0
        else f"Player has been steady in regular season"
    ),
    "past_game1_gs_avg": lambda v: (
        f"Player historically underperforms in Game 1s" if v > 0
        else f"Player historically delivers in Game 1s"
    ),
    "past_game7_gs_avg": lambda v: (
        f"Player historically underperforms in Game 7s" if v > 0
        else f"Player historically rises in Game 7s"
    ),
    "past_elim_gs_avg": lambda v: (
        f"Player has struggled in past elimination games" if v > 0
        else f"Player has performed well in past elimination games"
    ),
    "is_home": lambda v: (
        f"Away game adds risk" if v > 0 else f"Home court provides an advantage"
    ),
    "series_game_number": lambda v: (
        f"Later in the series, predictions get less stable" if v > 0
        else f"Early in the series, baseline conditions apply"
    ),
    "is_elimination_game": lambda v: (
        f"Elimination game pressure increases risk" if v > 0
        else f"No elimination pressure"
    ),
    "gs_in_series_so_far_mean": lambda v: (
        f"Player has underperformed earlier in this series" if v > 0
        else f"Player has been productive earlier in this series"
    ),
    "last_game_gs": lambda v: (
        f"Player had a weak previous game (momentum risk)" if v > 0
        else f"Player had a strong previous game (positive momentum)"
    ),
    "career_playoff_underperform_rate": lambda v: (
        f"Player has a history of fading in playoffs" if v > 0
        else f"Player has historically risen in playoffs"
    ),
    "days_rest": lambda v: (
        f"Less rest than usual" if v > 0 else f"Plenty of rest"
    ),
}


def _humanize(feature: str, shap_val: float) -> str:
    template = HUMANIZE.get(feature)
    if template is not None:
        return template(shap_val)
    # Fallback: generic message
    pretty = feature.replace("_z", "").replace("_", " ")
    direction = "↑" if shap_val > 0 else "↓"
    return f"{pretty.capitalize()} {direction} (contribution {shap_val:+.2f})"


class ModelExplainer:
    """Lazy SHAP wrapper compatible with XGBoost and sklearn ensemble models."""

    def __init__(self, model, feature_columns: list[str]):
        self.model = model
        self.feature_columns = feature_columns
        self._explainer = None  # lazy

    def _get_explainer(self):
        if self._explainer is None:
            try:
                import shap
                self._explainer = shap.TreeExplainer(self.model)
            except Exception:
                # If SHAP isn't available (e.g. lite container), fall back to
                # XGBoost's built-in feature importance.
                self._explainer = "fallback"
        return self._explainer

    def explain_one(self, x_row: pd.Series, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top-K contributing features with human-readable explanations."""
        explainer = self._get_explainer()
        if explainer == "fallback":
            return self._fallback_topk(x_row, top_k)

        X = x_row[self.feature_columns].to_frame().T.fillna(0)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):  # multiclass
            shap_values = shap_values[1]
        vals = np.asarray(shap_values)[0]

        # Pick top K by absolute SHAP value
        order = np.argsort(np.abs(vals))[::-1][:top_k]
        out = []
        for i in order:
            feat = self.feature_columns[i]
            v = float(vals[i])
            out.append({
                "feature": feat,
                "shap_value": v,
                "human_readable": _humanize(feat, v),
            })
        return out

    def _fallback_topk(self, x_row, top_k):
        try:
            importances = self.model.feature_importances_
        except AttributeError:
            return []
        order = np.argsort(importances)[::-1][:top_k]
        return [
            {"feature": self.feature_columns[i],
             "shap_value": float(importances[i]),
             "human_readable": _humanize(self.feature_columns[i], 1.0)}
            for i in order
        ]
