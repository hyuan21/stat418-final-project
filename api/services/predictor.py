"""Load the trained model from disk and run inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class Predictor:
    """Wraps a scikit-learn-compatible model with a stable predict_proba API."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._model = None  # lazy load

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Run `python -m src.models.train` first."
            )
        self._model = joblib.load(self.model_path)

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def predict_proba(self, x_row: pd.Series) -> float:
        """Return P(underperform=1) for one feature row."""
        X = x_row.to_frame().T.fillna(0)
        proba = self.model.predict_proba(X)[:, 1]
        return float(proba[0])

    def predict_expected_game_score(
        self, x_row: pd.Series, rs_avg: float | None,
        decline_threshold: float | None = None,
    ) -> float | None:
        """
        Convert the underperform probability into an EXPECTED Game Score.

        The model is a binary classifier, so we don't get a real regression
        target. We approximate by interpreting the probability as a mixture:
            P(underperform=1) * GS_under  +  P(underperform=0) * GS_over

        where GS_under is rs_avg minus the player's underperform threshold,
        and GS_over is rs_avg (a conservative "no drop" baseline).

        With a 20% relative threshold + 1.0 floor, this gives:
            expected_gs = rs_avg - p * threshold

        — which scales naturally with the player's baseline rather than the
        old flat 4.0 constant that produced the same -2.4 GS drop for every
        player regardless of role.
        """
        if rs_avg is None:
            return None
        p = self.predict_proba(x_row)
        threshold = decline_threshold if decline_threshold is not None else max(rs_avg * 0.2, 1.0)
        return float(rs_avg - p * threshold)


def classify_confidence(prob: float) -> str:
    """Map a probability to a low/medium/high confidence bucket."""
    distance_from_half = abs(prob - 0.5)
    if distance_from_half < 0.1:
        return "low"
    if distance_from_half < 0.25:
        return "medium"
    return "high"


# Classification threshold for binary underperform label.
#
# Empirical observation: the trained XGBoost model produces probabilities
# clustered in 0.30-0.45 for almost every superstar (RS_mean >= ~18). At the
# default 0.5 threshold, the model essentially never labels any superstar as
# "will underperform" — see scripts/analyze_stars.py output.
#
# Lowering the threshold to 0.40 brings the model's decision boundary closer
# to the mass of its actual probability distribution. This preserves the
# probability semantics (calibration) while making the binary label useful
# for high-baseline players. The probability itself is what the UI primarily
# shows; the label is just a convenience flag.
CLASSIFICATION_THRESHOLD = 0.40


def classify_label(prob: float, threshold: float = CLASSIFICATION_THRESHOLD) -> int:
    """Return the binary underperform label using the project's threshold."""
    return int(prob >= threshold)
