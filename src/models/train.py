"""
Train and compare three classifiers (Logistic Regression, Random Forest,
XGBoost) on the playoff feature table. Performs time-ordered train/val/test
split, Optuna-tuned XGBoost, and saves the best model + supporting artifacts.

Usage
-----
    python -m src.models.train \\
        --features data/processed/playoff_features.parquet \\
        --artifacts-dir api/artifacts \\
        --n-trials 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.utils.logging_config import get_logger

LOG = get_logger(__name__)


# --- Feature selection ---
# Modeling features are all "_z" columns plus a handful of raw context features.
RAW_CONTEXT_FEATURES = [
    "series_game_number", "is_home", "series_wins", "series_losses",
    "is_elimination_game", "is_closeout_game", "playoff_round",
    "days_rest", "career_playoff_games_played",
    "career_playoff_underperform_rate",
    "past_game1_gs_avg", "past_game7_gs_avg", "past_elim_gs_avg",
    "past_home_playoff_gs_avg", "past_away_playoff_gs_avg",
    "gs_in_series_so_far_mean", "last_game_gs", "last_game_underperformed",
]


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    z_cols = [c for c in df.columns if c.endswith("_z")]
    raw_cols = [c for c in RAW_CONTEXT_FEATURES if c in df.columns]
    return z_cols + raw_cols


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict time-ordered split: train ≤2020-21, val 2021-23, test ≥2023-24."""
    train = df[df["season"] <= "2020-21"]
    val = df[df["season"].isin(["2021-22", "2022-23"])]
    test = df[df["season"] >= "2023-24"]
    LOG.info("Train: %d rows, Val: %d rows, Test: %d rows",
             len(train), len(val), len(test))
    return train, val, test


def evaluate(model, X, y) -> dict:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.decision_function(X)
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def train_logistic(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, n_jobs=-1)),
    ])
    pipe.fit(X_train.fillna(0), y_train)
    return pipe


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    # RF can't handle NaN — impute with column means
    model.fit(X_train.fillna(X_train.mean(numeric_only=True)), y_train)
    return model


def tune_and_train_xgboost(
    X_train, y_train, n_trials: int = 50, seed: int = 42,
) -> xgb.XGBClassifier:
    """Optuna-tuned XGBoost using TimeSeriesSplit CV (with simplified search)."""
    try:
        import optuna
    except ImportError:
        LOG.warning("Optuna not installed; using default XGBoost params.")
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="auc", use_label_encoder=False, random_state=seed,
        )
        model.fit(X_train, y_train)
        return model

    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "eval_metric": "auc",
            "random_state": seed,
        }
        scores = []
        X_arr = X_train.values
        y_arr = y_train.values
        for tr_idx, va_idx in tscv.split(X_arr):
            m = xgb.XGBClassifier(**params)
            m.fit(X_arr[tr_idx], y_arr[tr_idx])
            proba = m.predict_proba(X_arr[va_idx])[:, 1]
            scores.append(roc_auc_score(y_arr[va_idx], proba))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    LOG.info("Best XGBoost AUC (CV): %.4f, params=%s",
             study.best_value, study.best_params)

    best = xgb.XGBClassifier(
        **study.best_params, eval_metric="auc", random_state=seed,
    )
    best.fit(X_train, y_train)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="data/processed/playoff_features.parquet")
    p.add_argument("--artifacts-dir", default="api/artifacts")
    p.add_argument("--n-trials", type=int, default=50)
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    feat_cols = select_feature_columns(df)
    LOG.info("Using %d feature columns.", len(feat_cols))

    train_df, val_df, test_df = time_split(df)
    X_train, y_train = train_df[feat_cols], train_df["y_underperform"]
    X_val, y_val = val_df[feat_cols], val_df["y_underperform"]
    X_test, y_test = test_df[feat_cols], test_df["y_underperform"]

    results = {}

    LOG.info("Training Logistic Regression baseline...")
    lr = train_logistic(X_train, y_train)
    results["logistic_regression"] = {
        "val": evaluate(lr, X_val.fillna(0), y_val),
        "test": evaluate(lr, X_test.fillna(0), y_test),
    }

    LOG.info("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)
    means = X_train.mean(numeric_only=True)
    results["random_forest"] = {
        "val": evaluate(rf, X_val.fillna(means), y_val),
        "test": evaluate(rf, X_test.fillna(means), y_test),
    }

    LOG.info("Training XGBoost with Optuna tuning (%d trials)...", args.n_trials)
    xgb_model = tune_and_train_xgboost(X_train, y_train, n_trials=args.n_trials)
    results["xgboost"] = {
        "val": evaluate(xgb_model, X_val, y_val),
        "test": evaluate(xgb_model, X_test, y_test),
    }

    LOG.info("Results summary:")
    for name, metrics in results.items():
        LOG.info("  %s — val AUC %.4f, test AUC %.4f",
                 name, metrics["val"]["roc_auc"], metrics["test"]["roc_auc"])

    # Save artifacts — XGBoost is the primary serving model.
    out = Path(args.artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, out / "model.pkl")
    joblib.dump(lr, out / "model_logistic.pkl")
    joblib.dump(rf, out / "model_random_forest.pkl")
    with open(out / "feature_columns.json", "w") as f:
        json.dump(feat_cols, f, indent=2)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    LOG.info("Artifacts written to %s", out)


if __name__ == "__main__":
    main()
