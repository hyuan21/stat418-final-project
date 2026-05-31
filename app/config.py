"""Streamlit App configuration. Reads from env vars with safe defaults."""
import os

# The Flask API base URL. In production (Streamlit Cloud) this is set
# to the Google Cloud Run URL via Streamlit's "Secrets" UI.
API_BASE_URL = os.environ.get("API_URL", "http://localhost:8080").rstrip("/")

# All endpoints live under /v1/
API_HEALTH = f"{API_BASE_URL}/v1/health"
API_PLAYERS = f"{API_BASE_URL}/v1/players"
API_TEAMS = f"{API_BASE_URL}/v1/teams"
API_PREDICT = f"{API_BASE_URL}/v1/predict"

# Request timeout in seconds
REQUEST_TIMEOUT = 15

# Display thresholds for the "underperform" color coding
HIGH_RISK_THRESHOLD = 0.65
LOW_RISK_THRESHOLD = 0.35
