# Running the Project on Your Local Machine

This guide walks through everything that needs to be done **on your computer** to take the codebase in this repo from "files only" to a fully deployed, working system.

The scaffolding is complete and 25 unit tests pass; the only remaining steps are:
1. Pulling real data from the NBA API (~7 hours, can run unattended)
2. Training the model on the real data (~10 min)
3. Deploying to Cloud Run + Streamlit Cloud

## 0. Prerequisites
- Python 3.10 or 3.11
- Docker Desktop running
- `gcloud` CLI installed and authenticated
- A GitHub account with this repo pushed up

## 1. Local setup

```bash
cd "C:\Users\Administrator\Desktop\stat418 final project\Stat 418 Final Project"
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

## 2. Verify the test suite passes locally

```bash
$env:PYTHONPATH = "."             # PowerShell
pytest tests/ -v
pytest api/tests/ -v
```

Expected: **25 passed**.

## 3. Scrape NBA data (~7 hours, run overnight)

```bash
python -m src.data.collect_game_logs --start-season 2003-04 --end-season 2024-25
```

Notes:
- The scraper caches every response to `data/raw/`. If your computer sleeps or the network drops, just rerun the same command — it picks up where it left off.
- Expect ~600 MB in `data/raw/` when complete.
- Sanity check: `data/raw/player_game_logs/regular_season/` should contain thousands of JSON files when finished.

## 4. Build the modeling dataset

```bash
python -m src.data.parse_raw       # Raw JSON → Parquet tables
python -m src.data.build_features  # Feature engineering with leakage checks
```

This should take ~5 minutes and produce `data/processed/playoff_features.parquet` with ~45,000 rows.

## 5. Train the model

```bash
python -m src.models.train --n-trials 50
```

This trains all three models (Logistic Regression, Random Forest, XGBoost with Optuna), evaluates on the held-out 2023–25 test set, and writes:
- `api/artifacts/model.pkl`
- `api/artifacts/feature_columns.json`
- `api/artifacts/metrics.json`

**Acceptance check:** Open `api/artifacts/metrics.json` and confirm XGBoost test `roc_auc ≥ 0.65`. If not, raise the underperform threshold in `src/data/build_features.py` (`UNDERPERFORM_THRESHOLD = 3.0`) and retrain.

## 6. Local end-to-end test

```bash
docker-compose up --build
# → http://localhost:8501 (App)
# → http://localhost:8080/docs (Swagger)
```

Pick a player, opponent, and game number; click Predict. You should see a probability, expected Game Score, and top-3 SHAP explanations.

## 7. Push to GitHub

```bash
git init
git remote add origin https://github.com/HanzhangYuan/stat418-final-project.git
git add .
git commit -m "Initial commit: complete pipeline + tests + docs"
git push -u origin main
```

GitHub Actions will run the test suite. Verify it goes green.

## 8. Deploy

See `docs/deployment.md` for the full deployment guide. The summary:

```bash
# API → Cloud Run
docker build -f api/Dockerfile -t gcr.io/YOUR_PROJECT_ID/nba-playoff-api .
docker push gcr.io/YOUR_PROJECT_ID/nba-playoff-api
gcloud run deploy nba-playoff-api \
    --image gcr.io/YOUR_PROJECT_ID/nba-playoff-api \
    --region us-central1 --memory 1Gi --allow-unauthenticated
```

Then go to [share.streamlit.io](https://share.streamlit.io), connect the repo, point at `app/streamlit_app.py`, and add `API_URL` to Secrets pointing at your Cloud Run URL.

## 9. Submit

* **June 1, 11:59 PM** — submit `presentations/final/hanzhang-yuan-final.pdf` via the course PR
* **June 2, EOD** — submit the GitHub repo URL via the course PR
* Keep services live through **June 9** for instructor evaluation

## Troubleshooting

**Scraper says "Too Many Requests"** → Increase `min_request_interval_seconds` in `src/data/nba_api_client.py` from 1.2 to 2.0 and rerun.

**Tests fail with `ModuleNotFoundError`** → Ensure `PYTHONPATH=.` is set when running pytest.

**Cloud Run timeout** → Increase `--timeout` from 60 to 120 seconds.

**Streamlit App can't reach API** → Confirm `API_URL` Secret on Streamlit Cloud has no trailing slash and includes `https://`.
