"""
Post-training analysis of the trained model. Produces:
  1. SHAP global feature-importance ranking
  2. Slice metrics (AUC by series_game_number, by playoff_round)
  3. 5 concrete prediction examples from 2024-25 playoffs

Run after `python -m src.models.train`:

    python scripts/analyze_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "playoff_features.parquet"
MODEL_PATH = ROOT / "api" / "artifacts" / "model.pkl"
FEATURE_COLS_PATH = ROOT / "api" / "artifacts" / "feature_columns.json"
LOGS_PATH = ROOT / "data" / "processed" / "player_game_logs.parquet"


def main() -> None:
    print("=" * 70)
    print("POST-TRAINING ANALYSIS")
    print("=" * 70)

    df = pd.read_parquet(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    feature_cols = json.loads(FEATURE_COLS_PATH.read_text())

    test = df[df["season"] >= "2023-24"].copy()
    X_test = test[feature_cols].fillna(0)
    y_test = test["y_underperform"]

    # ----- 1. SHAP global feature importance -----
    print("\n--- 1. SHAP Top 15 Global Feature Importance ---\n")
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        # Sample to keep this fast — 1000 rows is enough for ranking
        sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        top = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False).head(15)
        print(top.to_string())
    except Exception as e:
        # Fallback to XGBoost's built-in feature_importances_
        print(f"(SHAP failed: {e}; using built-in feature_importances_)")
        importances = model.feature_importances_
        top = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)
        print(top.to_string())

    # ----- 2. Slice metrics -----
    print("\n--- 2. Test ROC-AUC sliced by series_game_number ---\n")
    proba = model.predict_proba(X_test)[:, 1]
    test["_proba"] = proba
    for g in sorted(test["series_game_number"].unique()):
        slc = test[test["series_game_number"] == g]
        if len(slc) < 50 or slc["y_underperform"].nunique() < 2:
            print(f"  Game {int(g)}: n={len(slc):>5} (insufficient or no class variation)")
            continue
        auc = roc_auc_score(slc["y_underperform"], slc["_proba"])
        print(f"  Game {int(g)}: n={len(slc):>5}, AUC={auc:.4f}")

    print("\n--- 3. Test ROC-AUC sliced by playoff_round ---\n")
    for r in sorted(test["playoff_round"].unique()):
        slc = test[test["playoff_round"] == r]
        if len(slc) < 50 or slc["y_underperform"].nunique() < 2:
            print(f"  Round {int(r)}: n={len(slc):>5} (insufficient or no class variation)")
            continue
        auc = roc_auc_score(slc["y_underperform"], slc["_proba"])
        print(f"  Round {int(r)}: n={len(slc):>5}, AUC={auc:.4f}")

    print("\n--- 4. Test ROC-AUC sliced by player tier ---\n")
    q75 = df["rs_game_score_mean"].quantile(0.75)
    test["_tier"] = np.where(test["rs_game_score_mean"] >= q75, "Star", "Role")
    for tier in ["Star", "Role"]:
        slc = test[test["_tier"] == tier]
        if len(slc) < 50 or slc["y_underperform"].nunique() < 2:
            continue
        auc = roc_auc_score(slc["y_underperform"], slc["_proba"])
        print(f"  {tier:5s} (n={len(slc):>5}): AUC={auc:.4f}")

    # ----- 5. Concrete examples -----
    print("\n--- 5. Sample predictions on 2024-25 playoff games ---\n")
    print("Players sorted by RS_mean within recent season:")
    print()
    # Try to attach player_name from logs
    try:
        logs = pd.read_parquet(LOGS_PATH)
        name_map = (logs[["player_id"]].assign(
            player_name="player_" + logs["player_id"].astype(str)
        ).drop_duplicates("player_id").set_index("player_id")["player_name"])
        # Look for actual name columns
        # (If player_name was kept in logs, it would be here)
    except Exception:
        name_map = pd.Series(dtype=str)

    recent = test[test["season"] == "2024-25"].sort_values(
        "rs_game_score_mean", ascending=False
    )
    if recent.empty:
        recent = test.sort_values("rs_game_score_mean", ascending=False)

    cols_show = ["player_id", "season", "series_game_number",
                 "rs_game_score_mean", "game_score", "_proba", "y_underperform"]
    cols_show = [c for c in cols_show if c in recent.columns or c == "_proba"]
    print(recent[cols_show].head(10).to_string(index=False,
                                                float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("Analysis complete. Above results can be cited in writeup section 5.6.")
    print("=" * 70)


if __name__ == "__main__":
    main()
