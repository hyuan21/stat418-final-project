# API — Flask Model Service

Stateless REST API that serves the trained XGBoost model. Deployed to **Google Cloud Run**.

## Endpoints

All prediction/lookup endpoints are versioned under `/v1`. `/docs` and `/openapi.json` sit at the root for convenience.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/health` | Liveness probe (returns `{"status": "ok", "model_loaded": ..., "n_feature_columns": ...}`) |
| GET | `/v1/players?season=YYYY-YY` | List of player IDs/names supported by the model (optional `season` filter restricts to players who appeared in that season's playoffs) |
| GET | `/v1/teams` | List of team IDs/names with defensive stats available |
| GET | `/v1/matchups?player_id=...&season=YYYY-YY` | For a given player+season, return the opponents they actually faced in the playoffs and how many games each series went |
| GET | `/v1/actual_result?player_id=...&season=YYYY-YY&opponent_team_id=...&series_game_number=...` | Return the actual Game Score for a specific historical playoff game, for comparison vs. the model's prediction |
| POST | `/v1/predict` | **Main endpoint** — predict underperformance probability |
| GET | `/docs` | Swagger UI (auto-generated from flask-restx) |
| GET | `/openapi.json` | OpenAPI 3.0 schema |

## `POST /v1/predict`

**Request body:**
```json
{
  "player_id": "1629029",
  "season": "2024-25",
  "opponent_team_id": "1610612738",
  "series_game_number": 5,
  "is_home": true,
  "series_score": "2-2",
  "feature_overrides": {
    "pts_per_game": 35.0,
    "fg_pct": 0.50
  }
}
```

**Response (200):**
```json
{
  "prediction": {
    "underperform_probability": 0.67,
    "underperform_label": 1,
    "confidence": "medium"
  },
  "context": {
    "player_name": "Luka Doncic",
    "opponent_name": "Boston Celtics",
    "regular_season_gs_avg": 24.5,
    "decline_threshold_gs": 23.5
  },
  "explanation": {
    "top_factors": [
      {"feature": "opp_def_rating_z", "shap_value": 0.12,
       "human_readable": "Opponent defense is very strong (top 5 in NBA)"},
      {"feature": "past_game5_gs_avg", "shap_value": 0.08,
       "human_readable": "Player has historically underperformed in Game 5s"},
      {"feature": "is_home", "shap_value": -0.05,
       "human_readable": "Home court provides a small advantage"}
    ]
  },
  "similar_historical_games": [
    {
      "player": "Luka Doncic", "season": "2021-22",
      "opponent": "Golden State Warriors",
      "series_game_number": 5,
      "predicted_gs": 22.0, "actual_gs": 14.3
    }
  ]
}
```

**Error responses:** structured JSON with `error_code` and `message`; see `routes/predict.py` for the full list.

## Run locally
```bash
cd api
pip install -r requirements.txt
python app.py
# → http://localhost:8080
# Swagger UI → http://localhost:8080/docs
# Example: curl http://localhost:8080/v1/health
```

## Run with Docker
```bash
docker build -t nba-playoff-api .
docker run -p 8080:8080 nba-playoff-api
```

## Tests
```bash
cd api && pytest tests/ -v
```

## Deployment to Cloud Run
See `../.github/workflows/deploy.yml` for the automated pipeline. Manual:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/nba-playoff-api
gcloud run deploy nba-playoff-api \
    --image gcr.io/PROJECT_ID/nba-playoff-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```
