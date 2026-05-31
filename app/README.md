# App — Streamlit Frontend

Interactive web app for predicting NBA playoff underperformance. Deployed to **Streamlit Community Cloud**.

## What the user sees

1. **Player picker** — dropdown of all NBA players in the database (~3,500 options)
2. **Opponent picker** — dropdown of all 30 NBA teams
3. **Series game number** — slider 1 to 7
4. **Home / Away toggle**
5. **Current series score** — dropdown (`0-0`, `1-0`, ..., `3-3`)
6. **(Optional) Advanced mode** — expandable panel with sliders to override the player's regular-season stats for what-if analysis

On submit, the app POSTs to the API and renders:
- The underperformance probability as a large metric and progress bar
- The expected Game Score vs. the player's regular-season average
- **Top 3 SHAP factors** rendered in plain English ("Opponent defense is top 5 in NBA")
- A small table of historical games where this player faced similar conditions

## Run locally
```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
# → http://localhost:8501
```

## Configuration

The app reads `API_URL` from the environment (default: `http://localhost:8080`).

When deployed to Streamlit Cloud, set `API_URL` to the Cloud Run URL via the **Secrets** UI in the Streamlit dashboard.

## Run with Docker
```bash
docker build -t nba-playoff-app .
docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8080 nba-playoff-app
```

## Deployment to Streamlit Community Cloud
1. Push to GitHub
2. Visit https://share.streamlit.io and connect the repo
3. Point the app to `app/streamlit_app.py`
4. Add `API_URL` to Secrets
5. Deploy → public URL within ~1 minute
