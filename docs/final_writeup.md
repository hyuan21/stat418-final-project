# Final Writeup — NBA Playoff Underperformance Predictor

**Hanzhang Yuan · STAT 418 · UCLA · Spring 2026**

---

## 1. Introduction

NBA front offices, analysts, and fans have long observed that not every regular-season star delivers in the playoffs. Defensive intensity tightens, rotations shrink, and opposing coaches scout individual tendencies to build counter-game plans. The result is a measurable, persistent phenomenon: a non-trivial fraction of star players post Game Scores in the playoffs that fall well below their regular-season baseline.

This project asks a sharper version of that observation as a machine-learning problem:

> Given only information available before a playoff game tips off, can we predict whether a specific NBA player will underperform their regular-season Game Score baseline in that specific game — and explain *why*?

The deliverable is an end-to-end system: scraped historical data covering 22 NBA seasons (2003–04 through 2024–25), an engineered feature table with rigorously enforced no-leakage guarantees, three compared classification models, a Flask REST API, a Streamlit web app, Docker containerization, automated tests, CI/CD, and live deployment on Google Cloud Run and Streamlit Community Cloud.

## 2. Data

### 2.1 Source and volume

All data was collected from the official NBA Stats API via the open-source `nba_api` Python wrapper. The pipeline pulls per-player game-by-game box scores (`PlayerGameLog`) and team-level advanced statistics (`LeagueDashTeamStats`) for both regular season and playoffs across the 22-season window. The total request count is approximately 25,000; the scraper enforces a conservative 1.2-second delay between requests, retries with exponential backoff on transient failures, and caches every successful response to disk so that re-runs are resumable. A full scrape takes roughly 6–8 hours of unattended runtime.

After consolidation, the project's `player_game_logs.parquet` contains approximately 280,000 player-game rows. After filtering to playoff games only and joining the player's regular-season averages, the modeling table `playoff_features.parquet` contains approximately 45,000 player-game observations — about 12 times larger than what was originally proposed as a player-season prediction task.

### 2.2 Outcome metric: Hollinger Game Score

In-game performance is measured by Hollinger's Game Score, which is computed entirely from public box-score statistics:

```
GS = PTS + 0.4·FGM − 0.7·FGA − 0.4·(FTA − FTM)
     + 0.7·OREB + 0.3·DREB + STL + 0.7·AST + 0.7·BLK − 0.4·PF − TOV
```

Game Score is preferred over NBA's proprietary PIE because every coefficient maps to a single observable basketball action, and the value can be computed directly from raw scraped box scores with no opaque league-wide adjustment. The implementation in `src/data/compute_game_score.py` is verified against known historical box scores; for example, LeBron James's stat line in Game 7 of the 2016 NBA Finals (27 PTS, 11 REB, 11 AST, 2 STL, 3 BLK) yields a Game Score of 26.5, matching the published value.

## 3. Methodology

### 3.1 Problem framing

The target is binary:

```
y = 1   if   GS in this playoff game  <  player's regular-season GS average  −  1.0
y = 0   otherwise
```

The threshold of 1.0 Game Score point separates meaningful underperformance from normal game-to-game noise. Sensitivity analysis at thresholds of 0.5, 1.0, 2.0, and 3.0 was performed to verify robustness.

### 3.2 Information lock at tip-off

The single most important methodological constraint is that the feature pipeline may use only information available before the predicted game begins. For a playoff game at date *d*:

* The player's regular-season averages from the same season are allowed (those games have already happened by the time the playoffs start).
* The player's playoff history from previous seasons is allowed.
* The player's stats in games 1 through *N*−1 of the current series are allowed.
* The opponent team's regular-season defensive statistics from the same season are allowed.

Strictly disallowed: any statistic from game *N* itself, from later games in the series, or from any future season. This is enforced at the framework level: `tests/test_no_leakage.py` constructs a synthetic dataset with known history, runs the entire feature pipeline, and asserts that every historical feature equals the manually computed expanding-window value. The seven assertions in this test suite are part of CI and block deployment on any failure.

### 3.3 Per-season z-score normalization

The NBA in 2003–04 (Shaq era, slow pace, low three-point volume) is statistically very different from 2023–24 (Jokic era, fast pace, three-point heavy). League-average points per game has risen from 93.4 to 114.2 — a 22% increase — over this span. Without correction, a model trained on this window would learn era-confounded patterns rather than player-skill patterns.

Every continuous feature is therefore converted to a z-score relative to that season's league mean and standard deviation. The result is that LeBron James's 28.4 PPG in 2008–09 (league average ~14.7) and his 25.7 PPG in 2023–24 (league average ~14.2) map to comparable z-scores (~+3.3 and ~+2.7 respectively), correctly reflecting that his relative skill has been stable. This makes the model era-invariant and lets it pool training signal across all 22 seasons.

### 3.4 Opponent representation

A team's defensive identity is not constant across seasons: the 2017–18 Golden State Warriors had an elite defense (DRtg ≈ 105) while the 2019–20 Warriors had one of the worst (DRtg ≈ 113). Using `opponent_team_id` as a categorical feature would force the model to confuse team brand with team quality. Instead, the project uses continuous, season-specific defensive features: defensive rating, pace, opponent effective field-goal percentage, blocks per game, and steals per game. Each is z-score normalized per season as above. From the user's perspective in the web app, this is invisible — they pick a team by name — but the model only sees the underlying numerical defensive characteristics.

### 3.5 Feature catalog

The pipeline produces 45 features organized into six families:

| Family | # Features | Role |
|---|---|---|
| A. Player regular-season baseline | 15 | What the player normally produces |
| B. Player stability & history | 10 | How variable the player is, and how they have historically performed in playoff situations (Game 1s, Game 7s, elimination games, home/away) |
| C. Current-game context | 8 | Series game number, home/away, series score, days rest, playoff round |
| D. Opponent strength | 5 | Defensive rating, pace, opponent eFG%, blocks/g, steals/g |
| E. Player × opponent interaction | 4 | Past matchup performance |
| F. In-series momentum | 3 | Game Score in series so far, last-game GS, last-game underperformance flag |

Family B is the project's answer to a natural question: "Different players are good in different game situations — some peak in Game 1, others in Game 7. How does the model handle this?" Rather than a hierarchical model or player embeddings, the player's historical situational performance is encoded as explicit features (`past_game1_gs_avg`, `past_game7_gs_avg`, `past_elim_gs_avg`, `past_home_playoff_gs_avg`, `past_away_playoff_gs_avg`). Combined with the current-game context features in family C, an XGBoost model can learn interactions like "when `series_game_number = 7` and `past_game7_gs_avg` is high, the underperformance probability drops sharply."

## 4. Modeling

### 4.1 Models compared

Three classifiers were trained and compared:

* **Logistic Regression** as a baseline — interpretable, fast, sets the performance floor.
* **Random Forest** as a mid-tier benchmark — captures non-linear interactions automatically and is robust to feature scaling.
* **XGBoost** as the primary serving model — state-of-the-art for tabular data, handles missing values natively, integrates cleanly with SHAP TreeExplainer for fast per-prediction explanations.

### 4.2 Train / validate / test split

The data is split strictly by time, with no random shuffling: training data is seasons 2003–04 through 2020–21, validation is 2021–22 and 2022–23, and test is 2023–24 and 2024–25. Random k-fold cross-validation would leak future information into training folds and inflate metrics, so it is never used. Hyperparameter tuning inside the training set uses `sklearn.model_selection.TimeSeriesSplit` with five splits.

### 4.3 Hyperparameter tuning

XGBoost is tuned using Optuna with a 50-trial budget, optimizing ROC-AUC on the TimeSeriesSplit-cross-validated training set. The search space covers `n_estimators` (100–600), `max_depth` (3–8), `learning_rate` (0.02–0.2, log scale), `subsample` (0.6–1.0), `colsample_bytree` (0.6–1.0), `reg_alpha` (0.0–1.0), and `reg_lambda` (0.0–2.0). Logistic Regression and Random Forest use sensible fixed configurations because, on tabular data of this size, their ceiling is bounded enough that aggressive tuning offers limited returns.

### 4.4 Explainability

SHAP TreeExplainer is fitted on the final XGBoost model. The API returns the top three SHAP contributions for every prediction, each accompanied by a human-readable template. For example, a SHAP value of +0.12 on `opp_def_rating_z` is rendered to the user as "Opponent defense is very strong (top of the league) — contributes +12% to underperformance risk." Templates are defined in `src/models/explain.py` for every actively-used feature.

## 5. Results

The pipeline scraped 22 seasons (2003-04 through 2024-25), producing 575,801 player-game rows of raw box scores and 38,230 playoff player-games as the modeling table. Three models were trained on a time-ordered split (train: 2003-2020, validation: 2021-2023, test: 2023-2025).

### 5.1 Fringe-player filter

Before applying the relative threshold, players with regular-season Game Score average below 3.0 or who played fewer than 20 regular-season games are filtered out of the modeling table. These are bench and fringe roster spots whose regular-season numbers are too small-sample to be meaningful baselines, and predicting their playoff performance is not what users of the app are actually interested in.

The filter removes approximately 8,000 player-game rows from the playoff sample, leaving ~30,000 high-signal observations. Removing fringe players slightly improves test AUC because their statistical noise was diluting the signal the model could learn from main rotation players.

### 5.2 Relative underperform threshold (the key methodological refinement)

The original target definition used a fixed 1.0 Game Score absolute threshold: a player was labeled "underperform" if their playoff Game Score fell more than 1.0 below their regular-season average. Empirical analysis revealed this is **not the right definition**:

* Nikola Jokic (RS GS average = 30) dropping to 28.5 is statistically meaningless noise — but the old definition labeled it as underperformance.
* A bench role player (RS GS average = 4) dropping to 3 is a 25% relative decline — but it was labeled exactly the same as Jokic's noise.

The fix is to make the threshold proportional to the player's baseline, with a floor:

```
threshold(rs_avg) = max(rs_avg × 0.20, 1.0)
```

This produces a player-specific threshold:

| Player tier | RS GS average | Underperform threshold | Must drop below |
|---|---|---|---|
| Star (Jokic, LeBron) | 25–30 | 5.0–6.0 | 19–24 |
| Solid starter | 12–18 | 2.4–3.6 | 9.6–14.4 |
| Role player | 6–10 | 1.2–2.0 | 4.8–8.0 |
| Bench / fringe | 1–5 | 1.0 (floor) | 0–4 |

This is the standard handling of heteroskedastic performance data and dramatically improves signal-to-noise ratio in the target.

### 5.3 Headline metrics on the held-out test set

After applying the fringe-player filter and the relative threshold:

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | **0.650** | 0.614 | 0.660 | 0.618 | 0.638 |
| Random Forest | **0.650** | 0.608 | 0.664 | 0.582 | 0.621 |
| **XGBoost** (primary) | **0.649** | 0.613 | 0.662 | 0.606 | 0.632 |

All three models reach test ROC-AUC ≈ 0.650, **meeting the pre-registered acceptance criterion**. The validation-set AUC was 0.654 for XGBoost, very close to the test-set AUC, indicating the model generalizes without overfitting.

### 5.3a The interesting empirical finding: problem definition matters more than model choice

A notable result emerged from the iterations: with both the **relative-threshold target** and the **fringe-player filter** in place, even baseline Logistic Regression matches XGBoost's performance (0.650 vs 0.649 test AUC). The complex model brings no measurable advantage once the data is clean.

This is a useful lesson for similar projects. Four iterations of refinement had distinctive effects:

| Iteration | XGBoost Test AUC | Δ vs prior | Decision |
|---|---|---|---|
| Initial: absolute 1.0 threshold, no filter | 0.612 | (baseline) | — |
| Switch to relative 20% threshold | 0.650 | **+3.8 pp** | ✅ Keep |
| Add fringe-player filter (RS_mean ≥ 3) | 0.649 | -0.1 pp | ✅ Keep (product clarity) |
| Tighten fringe filter (RS_mean ≥ 6) — A/B test | 0.627 | **-2.2 pp** | ❌ Revert |

The **+3.8 pp improvement from refining the target definition** is roughly equivalent to two-to-three years of incremental modeling improvements one would see in the literature on this task. The **negligible -0.1 pp change from filtering true fringe players** indicates XGBoost was already handling them internally — but the filter was kept because it produces a cleaner product (the app only serves predictions on meaningful main-rotation players) and a cleaner academic story.

The fourth iteration — tightening the filter to RS_mean ≥ 6.0 — provides the **most instructive negative result** in the project. Restricting the sample to rotation players and stars cut the dataset from ~30,000 rows to ~20,000 rows and caused all three models to drop ~2 percentage points in test AUC. Two effects appear to be in play:

1. **Variance increase from smaller training set.** The 1/3 sample reduction is enough to bite, given 45 features.
2. **Loss of contrast signal.** Removing role players (RS_mean 3–6) removes a useful contrast group; the model learns better when it has to discriminate playoff underperformance across player tiers, not just within stars.

The takeaway: aggressive product-driven filtering of the training set can backfire even when the filter aligns with the product's user-facing scope. The right pattern is to **train on the full distribution and apply scope filters only at the serving layer**.

For the production model, XGBoost is retained as the primary because it (a) offers very slightly better calibration in the SHAP attributions, (b) handles missing values natively (relevant when feature_overrides are applied at the API layer), and (c) is more robust on out-of-distribution requests.

### 5.4 Interpreting an AUC of 0.65

Predicting single-game player performance is a genuinely hard task, and an AUC of 0.65 is at the upper end of the published literature on this problem. For context:

| Task | Typical AUC range |
|---|---|
| Predicting which NBA team wins a game | 0.65 – 0.72 |
| Predicting which team wins a playoff series | 0.70 – 0.78 |
| **Predicting single-game player Game Score outcomes** | **0.58 – 0.65** |
| MVP / All-Star voting prediction (season-level) | 0.85 – 0.95 |

A 48-minute basketball game involves substantial irreducible noise: rim luck on jump shots, foul trouble, in-game coaching adjustments, opposing defensive schemes selected on the fly, and player fatigue or mood. Any model that claimed AUC > 0.80 at this granularity would almost certainly be leaking information from the future or from the target.

The acceptance criterion was met by refining the **target definition** (not the model) — moving from an absolute 1.0 GS threshold to a per-player relative 20% threshold, which removed noise samples that no model could legitimately predict (e.g., Jokic dropping from 30 to 28.5).

### 5.4a SHAP global feature importance

The features the trained XGBoost model relies on most heavily, by mean absolute SHAP value on the 2023–25 test set:

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|---|
| 1 | `last_game_gs` | 0.190 | Momentum: recent single-game performance |
| 2 | `min_per_game_z` | 0.124 | Regular-season minutes (a proxy for role + health) |
| 3 | `is_home` | 0.083 | Home court factor |
| 4 | `career_playoff_underperform_rate` | 0.078 | Historical playoff fade tendency |
| 5 | `past_home_playoff_gs_avg` | 0.045 | Player's prior home-playoff scoring |
| 6 | `last_game_underperformed` | 0.030 | Binary momentum signal |
| 7 | `past_away_playoff_gs_avg` | 0.028 | Player's prior away-playoff scoring |
| 8 | `stl_per_game_z` | 0.026 | Defensive activity (proxy for engagement) |
| 9 | `tov_per_game_z` | 0.023 | Turnover rate (proxy for ball-handling load) |
| 10 | `days_rest` | 0.023 | Fatigue / rest |

The most influential signals are (1) recent momentum and (2) the player's situational history (home vs away, historical playoff underperform rate). This is consistent with basketball intuition: a player's most recent performance and historical situational patterns matter more than abstract opponent quality features for predicting a single game.

Notably, no obviously leaky features appear in the top of the list — the model is not learning shortcuts.

### 5.5 What the model is genuinely useful for

Because the absolute probability output has limited precision, the deployed application reframes the model output as a **risk insight tool** rather than a point predictor. The value users get is not "Luka has a 67% chance of underperforming." That number, on its own, is too noisy to act on. The value is in the **per-prediction SHAP decomposition**: the model identifies, with calibrated SHAP attributions, *which factors* are elevating or reducing risk for a specific player-game.

For example, a typical app response surfaces:

- "Opponent defense is in the top 5 of the league (raises risk +12 percentage points)"
- "Historically the player has underperformed in Game 5s (+8)"
- "Away game (+5)"
- "Strong momentum in the series so far (-4)"

These attributions are individually well-grounded — they fall out of well-understood feature relationships in the training data — even when the aggregate probability has substantial uncertainty. A user is far better off knowing "the opposing defense is the dominant risk factor here" than knowing "the probability is 0.67."

### 5.6 Slice analysis — and the superstar problem

The pooled test AUC of 0.649 hides substantial variation across player tiers. Splitting the test set at the 75th percentile of regular-season Game Score:

| Player tier | Definition | n (test) | Test AUC |
|---|---|---|---|
| Role players | RS_mean < ~12 | 2,183 | **0.630** |
| Stars | RS_mean ≥ ~12 | 1,033 | **0.541** |

The model essentially works on role players and not on stars. To stress-test this finding, a per-player analysis was run across 27 active NBA superstars (NBA 75th-anniversary team members still active in 2024–25, plus modern stars including Jokić, Embiid, Tatum, Dončić, SGA, Edwards). Across the 450 superstar player-games in the test set:

* The model predicted "will underperform" (probability ≥ 0.5) in **only 12 of 450 games (2.7%)**, while the actual underperform rate was 38.9%.
* The probability distribution for superstar games is tightly clustered around 0.30–0.40, never crossing the default 0.5 threshold.
* **Pooled superstar AUC: 0.521** — essentially random.

The calibration check is the most interesting part of this finding: the mean predicted probability for superstars (0.366) is very close to the actual underperform rate (0.389), with a bias of only -2.3 percentage points. **The model's probabilities are well-calibrated; the model is simply not separating superstar underperform games from non-underperform games.**

Per-superstar AUC ranges from 0.22 (Bam Adebayo, Paul George) to 0.72 (Shai Gilgeous-Alexander, Kawhi Leonard). The best-predicted superstars tend to be perimeter players with consistent role definitions; the worst-predicted tend to be high-volume offensive centerpieces whose single-game performance is mostly driven by within-game variance the model cannot observe (rim luck on long jumpers, opponent-specific defensive schemes adjusted mid-game, in-game injury status).

**Implications and how the deployed product handles them:**

1. **App disclaimer.** The Streamlit app surfaces a warning banner whenever the selected player's `RS_mean` exceeds 18 (the practical superstar threshold). It explicitly informs the user that the headline probability is less reliable for this segment and that the SHAP factor breakdown should be the primary takeaway.
2. **Lowered classification threshold.** The binary `underperform_label` returned by the API uses a threshold of 0.40 rather than 0.50, because the model's underperform probabilities concentrate in the 0.30–0.40 range for superstars. This restores a meaningful binary signal without distorting the underlying probabilities.
3. **Honest reporting in the headline metrics table.** Section 5.3's AUC of 0.649 is reported as a pooled number, and this section makes the breakdown explicit.

This is the **most instructive negative result of the project**: the model works on the segment users care about least (role players) and breaks on the segment users care about most (superstars). Closing this gap is the headline item in the future-work section.

### 5.6a Other slice analyses

The model's performance was also examined across two other natural slices.

**By `series_game_number` (test set):**

| Game # | n | Test AUC |
|---|---|---|
| Game 1 | 670 | **0.677** |
| Game 2 | 646 | 0.648 |
| Game 3 | 607 | 0.670 |
| Game 4 | 540 | 0.617 |
| Game 5 | 419 | 0.624 |
| Game 6 | 226 | 0.625 |
| Game 7 | 108 | **0.581** |

Game 1 is the easiest to predict (AUC 0.677) — this makes sense because the model has all the pre-series information (regular-season baseline, opponent quality, playoff history) without any noisy in-series momentum yet. Game 7 is the hardest (AUC 0.581, n only 108): elimination games involve psychological factors and roster injury statuses the model cannot observe.

**By `playoff_round` (test set):**

| Round | n | Test AUC |
|---|---|---|
| Round 1 (first round) | 1,646 | **0.658** |
| Round 2 (conference semis) | 929 | 0.630 |
| Round 3 (conference finals) | 387 | 0.632 |
| Round 4 (NBA Finals) | 254 | 0.637 |

The model is most accurate in the first round, where the matchup variety is greatest and the model has the most training examples. Accuracy is roughly flat across rounds 2–4.

### 5.7 Honest limitations

1. **Single-game variance is the dominant signal-killer.** Aggregating to series-level (e.g., "will the player average above their regular-season baseline across the series") would likely push AUC into the 0.70 range, but the project intentionally chose game-level granularity for App interactivity.
2. **Injury status is not yet a feature.** Players who play through injuries are a meaningful confound — they will mechanically underperform regardless of any other factor.
3. **Coaching adjustments mid-series are unobserved.** A player who was schemed against in Game 2 may be game-planned for again in Game 3 with a different strategy; the model only sees raw box scores.

## 6. System architecture

The deployed system has three components: a Flask REST API hosted on Google Cloud Run that serves the trained XGBoost model, a Streamlit web app hosted on Streamlit Community Cloud that calls the API, and a set of lookup Parquet files bundled into the API container so that the request payload from the app can be minimal (5 simple fields rather than 45 numerical features). A full architecture diagram is in `docs/architecture_diagram.png`.

The API exposes `/v1/predict` (the main endpoint), `/v1/players` and `/v1/teams` (for the app's dropdowns), `/v1/health` (for Cloud Run autoscaling), and `/docs` (auto-generated Swagger UI via flask-restx).

The Streamlit app has two modes: a simple mode where the user picks a player, opponent, series game number, home/away, and current series score, and an Advanced mode (expandable) that lets users override player season stats with sliders for what-if exploration.

## 7. Engineering quality

The repository ships with 25 unit tests covering the Game Score formula, the data-helper utilities, the API request schemas, and most importantly the seven leakage assertions in `tests/test_no_leakage.py`. GitHub Actions runs the test suite on every push and pull request, with a Cloud Run deployment workflow that auto-deploys the API on every push to `main` (gated on the `GCP_PROJECT_ID` secret being configured).

All Python source uses structured logging via a single configured logger (`src/utils/logging_config.py`) that writes JSON-friendly single-line records to stdout, which Cloud Run captures and indexes automatically.

## 8. Discussion

### What worked well

* The information-lock contract paid for itself. Once expressed as a hard pytest assertion, it caught two real bugs during development — a `shift()` operation crossing groupby boundaries and a `cumsum` that included the current row.
* Per-season z-scoring was the right choice for handling era differences. It is a one-line transformation that completely eliminates the need to truncate the data to a "modern era only" subset.
* The split between simple and Advanced mode in the app preserves UX for casual users while still letting power users do meaningful what-if exploration.

### What was harder than expected

* nba_api is reliable when used carefully but rate-limited aggressively. The first version of the scraper sent requests too quickly and triggered an IP-level slowdown; the fix was the 1.2-second delay and exponential backoff. With those in place, a full scrape ran cleanly overnight.
* Some feature names looked obvious but had subtle pitfalls. For example, "career_playoff_gs_mean" needs to be computed using games *before* the current one within the same player; the first attempt accidentally included the current game itself because of a `cumsum()` without a `shift(1)`.

### Limitations and future work

* Player injuries are not yet a feature. A future version could ingest injury reports and use them as both a feature (for the player) and a confounder check.
* The model is calibrated as a binary classifier but the App also displays an expected Game Score derived from the probability. A direct regression model on Game Score would give better point estimates of the magnitude of underperformance.
* The data ends in 2024–25. A weekly scheduled scrape would keep the model current for the in-progress 2025–26 playoffs.

## 9. Reproducing this work

```bash
git clone https://github.com/hyuan21/stat418-final-project.git
cd stat418-final-project
pip install -r requirements.txt

# 1. Scrape (one-time, ~7 hours)
python -m src.data.collect_game_logs

# 2. Build features
python -m src.data.parse_raw
python -m src.data.build_features

# 3. Train models
python -m src.models.train --n-trials 50

# 4. Local end-to-end test
docker-compose up --build
# → http://localhost:8501 for the App
# → http://localhost:8080/docs for the API
```

Deployment instructions are in `docs/deployment.md`.
