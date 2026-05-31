"""
Analyze model accuracy for each active NBA superstar individually.

Lists ~23 current-era superstars (NBA 75 Greatest still active + modern stars)
and computes:
  * how many playoff games they have in our test split (2023-2025)
  * AUC of the model's prediction on those games
  * accuracy at threshold 0.5
  * mean predicted probability vs actual underperform rate (calibration)
  * a per-game prediction table
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "playoff_features.parquet"
MODEL_PATH = ROOT / "api" / "artifacts" / "model.pkl"
FEATURE_COLS_PATH = ROOT / "api" / "artifacts" / "feature_columns.json"


# Active superstars: NBA 75 still playing + modern era top names.
SUPERSTARS = [
    # NBA 75 Greatest, still active in 2024-25
    ("LeBron James",            2544),
    ("Stephen Curry",           201939),
    ("Kevin Durant",            201142),
    ("James Harden",            201935),
    ("Russell Westbrook",       201566),
    ("Chris Paul",              101108),
    ("Kawhi Leonard",           202695),
    ("Anthony Davis",           203076),
    ("Damian Lillard",          203081),
    ("Giannis Antetokounmpo",   203507),
    ("Kyrie Irving",            202681),
    ("Paul George",             202331),
    ("Klay Thompson",           202691),
    ("Jimmy Butler",            202710),
    # Modern superstars not on the original 75 list
    ("Nikola Jokic",            203999),
    ("Joel Embiid",             203954),
    ("Jayson Tatum",            1628369),
    ("Luka Doncic",             1629029),
    ("Devin Booker",            1626164),
    ("Donovan Mitchell",        1628378),
    ("Trae Young",              1629027),
    ("Bradley Beal",            203078),
    ("De'Aaron Fox",            1628368),
    ("Shai Gilgeous-Alexander", 1628983),
    ("Anthony Edwards",         1630162),
    ("Karl-Anthony Towns",      1626157),
    ("Pascal Siakam",           1627783),
    ("Bam Adebayo",             1628389),
    ("Jaylen Brown",            1627759),
    ("Ja Morant",               1629630),
    ("Zion Williamson",         1629627),
    ("Tyrese Haliburton",       1630169),
]


def main() -> None:
    print("=" * 90)
    print("PER-SUPERSTAR PREDICTION ANALYSIS (Test split: 2023-24 & 2024-25)")
    print("=" * 90)

    df = pd.read_parquet(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    feature_cols = json.loads(FEATURE_COLS_PATH.read_text())

    test = df[df["season"] >= "2023-24"].copy()
    X_test = test[feature_cols].fillna(0)
    test["_proba"] = model.predict_proba(X_test)[:, 1]
    test["_pred"] = (test["_proba"] >= 0.5).astype(int)

    print()
    print(f"{'Player':<26}{'n':>5}{'RS_mean':>10}{'underperf':>11}{'pred>=.5':>10}{'mean_proba':>12}{'AUC':>8}{'Acc':>8}")
    print(f"{'-'*26}{'-'*5}{'-'*10}{'-'*11}{'-'*10}{'-'*12}{'-'*8}{'-'*8}")

    rows = []
    for name, pid in SUPERSTARS:
        slc = test[test["player_id"] == pid]
        n = len(slc)
        if n == 0:
            print(f"{name:<26}{0:>5}{'--':>10}{'--':>11}{'--':>10}{'--':>12}{'--':>8}{'--':>8}")
            continue
        rs_mean = slc["rs_game_score_mean"].mean()
        underperf_rate = slc["y_underperform"].mean()
        pred_rate = slc["_pred"].mean()
        mean_proba = slc["_proba"].mean()
        if slc["y_underperform"].nunique() == 2:
            auc = roc_auc_score(slc["y_underperform"], slc["_proba"])
        else:
            auc = float("nan")
        acc = accuracy_score(slc["y_underperform"], slc["_pred"])

        rows.append({
            "name": name, "n": n, "rs_mean": rs_mean,
            "underperf_rate": underperf_rate, "pred_rate": pred_rate,
            "mean_proba": mean_proba, "auc": auc, "acc": acc,
        })

        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
        print(
            f"{name:<26}{n:>5}{rs_mean:>10.2f}{underperf_rate:>11.2%}"
            f"{pred_rate:>10.2%}{mean_proba:>12.3f}{auc_str:>8}{acc:>8.2%}"
        )

    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    if rows:
        df_rows = pd.DataFrame(rows)
        present = df_rows[df_rows["n"] >= 3]
        print(f"\nSuperstars with >= 3 playoff games in test set: {len(present)}")
        print(f"Total playoff games covered:                   {present['n'].sum()}")

        # Pool all star games together for an aggregate AUC
        pooled = test[test["player_id"].isin([s[1] for s in SUPERSTARS])]
        if len(pooled) >= 50 and pooled["y_underperform"].nunique() == 2:
            pooled_auc = roc_auc_score(pooled["y_underperform"], pooled["_proba"])
            pooled_acc = accuracy_score(
                pooled["y_underperform"], (pooled["_proba"] >= 0.5).astype(int)
            )
            print(f"\nPooled across all superstars:")
            print(f"  n            = {len(pooled)}")
            print(f"  AUC          = {pooled_auc:.4f}")
            print(f"  Accuracy     = {pooled_acc:.4f}")
            print(f"  Mean proba   = {pooled['_proba'].mean():.4f}")
            print(f"  Actual rate  = {pooled['y_underperform'].mean():.4f}")

        # Calibration: does the average predicted probability match actual rate?
        print(f"\nCalibration on superstars:")
        print(f"  Average predicted probability: {pooled['_proba'].mean():.4f}")
        print(f"  Actual underperform rate:      {pooled['y_underperform'].mean():.4f}")
        bias = pooled["_proba"].mean() - pooled["y_underperform"].mean()
        print(f"  Bias (pred - actual):          {bias:+.4f}")
        if abs(bias) > 0.1:
            print(f"  ⚠️  Model is significantly {'over' if bias > 0 else 'under'}-predicting underperformance")

        # Players sorted by AUC for star-by-star ranking
        ranked = df_rows[df_rows["n"] >= 5].dropna(subset=["auc"])
        ranked = ranked.sort_values("auc", ascending=False)
        if len(ranked) >= 3:
            print(f"\nBest predicted superstars (n >= 5, sorted by AUC):")
            for _, r in ranked.head(5).iterrows():
                print(f"  {r['name']:<26} AUC={r['auc']:.3f}  n={int(r['n'])}")
            print(f"\nWorst predicted superstars (n >= 5, sorted by AUC):")
            for _, r in ranked.tail(5).iterrows():
                print(f"  {r['name']:<26} AUC={r['auc']:.3f}  n={int(r['n'])}")


if __name__ == "__main__":
    main()
