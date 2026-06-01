"""
Streamlit app — NBA Playoff Underperformance Predictor.

Two modes:
  1. Replay Mode — pick an actual 2023-24 / 2024-25 playoff game; the App
     shows model prediction vs the actual result.
  2. What-if Mode — pick any (player, opponent, game number) combination,
     including hypothetical matchups; the App shows model prediction only
     (no real result to compare to).

Run locally:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

from config import (
    API_HEALTH, API_PLAYERS, API_TEAMS, API_PREDICT, REQUEST_TIMEOUT,
    HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD, API_BASE_URL,
)


# Two season lists for the two modes.
# Replay needs a finished season (with full playoff data) so the App can
# show actual results next to model predictions. 2025-26 playoffs are
# still in progress and only partially scraped, so Replay restricts to
# 2023-24 and 2024-25.
# What-if uses only the player's regular-season baseline to predict a
# hypothetical / not-yet-played game, so it can use 2025-26's
# regular-season data even while the playoffs are in progress.
AVAILABLE_SEASONS_REPLAY = ["2024-25", "2023-24"]
AVAILABLE_SEASONS_WHATIF = ["2025-26", "2024-25", "2023-24"]


# ============ Page setup ============
st.set_page_config(
    page_title="NBA Playoff Underperformance Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      h1 { color: #1E3A8A; }
      .stMetric label { color: #64748B; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏀 NBA Playoff Underperformance Predictor")
st.caption(
    "An XGBoost model predicts whether an NBA player will underperform "
    "their regular-season Game Score in a specific playoff game, using only "
    "information available _before_ tip-off. "
    "Trained on 22 seasons (2003–2025). Test AUC: 0.65."
)


# ============ HTTP helpers ============
@st.cache_data(ttl=600, show_spinner=False)
def fetch_players(season: str | None = None) -> list[dict]:
    try:
        params = {"season": season} if season else {}
        r = requests.get(API_PLAYERS, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json().get("players", [])
    except Exception as e:
        st.warning(f"Could not load players from API: {e}")
        return []


@st.cache_data(ttl=600, show_spinner=False)
def fetch_teams() -> list[dict]:
    try:
        r = requests.get(API_TEAMS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json().get("teams", [])
    except Exception as e:
        st.warning(f"Could not load teams from API: {e}")
        return []


@st.cache_data(ttl=600, show_spinner=False)
def fetch_matchups(player_id: int, season: str) -> list[dict]:
    try:
        r = requests.get(
            f"{API_BASE_URL}/v1/matchups",
            params={"player_id": player_id, "season": season},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("matchups", [])
    except Exception:
        return []


def fetch_actual(player_id: int, season: str,
                  opponent_team_id: int, series_game_number: int) -> dict | None:
    try:
        r = requests.get(
            f"{API_BASE_URL}/v1/actual_result",
            params={
                "player_id": player_id, "season": season,
                "opponent_team_id": opponent_team_id,
                "series_game_number": series_game_number,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code >= 400:
            return None
        return r.json()
    except Exception:
        return None


def call_predict(payload: dict) -> dict[str, Any] | None:
    try:
        r = requests.post(API_PREDICT, json=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            st.error(f"API error {r.status_code}: {r.text}")
            return None
        return r.json()
    except requests.RequestException as e:
        st.error(f"Could not reach API: {e}")
        return None


# ============ Sidebar: mode selection + inputs ============
with st.sidebar:
    mode = st.radio(
        "Mode",
        ["🎬 Replay (real game)", "💡 What-if / Predict the future"],
        index=0,
        help=(
            "Replay: pick a real 2023–26 playoff game and see model "
            "prediction vs actual result.\n\n"
            "What-if: pick any player + any opponent + any game number. "
            "Use this to predict ongoing 2025-26 playoff games before they "
            "happen, or to ask hypothetical questions like 'how would Jokic "
            "do against the 1996 Bulls'."
        ),
    )
    is_replay = mode.startswith("🎬")

    st.divider()
    st.header("1. Pick a matchup")

    if is_replay:
        # --- Replay Mode: cascading dropdowns from real data ---
        season = st.selectbox("Season", AVAILABLE_SEASONS_REPLAY, index=0)
        st.caption(
            "ℹ️ The 2025-26 playoffs are still in progress, so Replay mode "
            "only supports completed seasons (2023-24 and 2024-25). "
            "To predict 2025-26 playoff games, switch to **What-if mode**."
        )

        players = fetch_players(season=season)
        if not players:
            st.warning(
                "No players returned by the API. Wait a few seconds for the "
                "API to finish starting up, then refresh the page."
            )
            st.stop()

        # Default to Jokic if present
        default_idx = 0
        for i, p in enumerate(players):
            if "Jokic" in str(p.get("player_name", "")):
                default_idx = i
                break

        player = st.selectbox(
            "Player",
            players,
            index=default_idx,
            format_func=lambda p: (
                f"{p.get('player_name', p.get('player_id'))} "
                f"(RS GS {p.get('rs_game_score_mean', 0):.1f})"
            ),
            help="Players who actually played in this season's playoffs, "
                 "sorted by regular-season Game Score.",
        )

        matchups = fetch_matchups(player["player_id"], season)
        if not matchups:
            st.warning(
                f"{player.get('player_name', 'This player')} had no playoff "
                f"games in {season}. Pick a different player."
            )
            st.stop()

        opponent = st.selectbox(
            "Opponent (actual playoff opponent)",
            matchups,
            format_func=lambda m: (
                f"{m.get('opponent_team_name', m.get('opponent_abbr'))} "
                f"— series went {m.get('n_games', '?')} games"
            ),
        )
        max_game = opponent.get("n_games", 7)
        game_num = st.slider(
            "Series game number",
            1, max_game, 1,
            help=f"This series went {max_game} games.",
        )
        opponent_team_id = int(opponent.get("opponent_team_id", 0))
        opponent_name = opponent.get("opponent_team_name", "Opp")
    else:
        # --- What-if Mode: free-form ---
        season = st.selectbox(
            "Season (for the player's regular-season stats)",
            AVAILABLE_SEASONS_WHATIF, index=0,
            help="The model uses this season's regular-season averages as the "
                 "player's baseline.",
        )
        if season == "2025-26":
            st.caption(
                "🔮 **Predicting the future.** You're using 2025-26 "
                "regular-season stats to predict a playoff game that may "
                "not have happened yet."
            )

        # Show ALL players (not restricted to playoff participants)
        players = fetch_players(season=None)
        if not players:
            st.warning(
                "No players returned by the API. Wait a few seconds for the "
                "API to finish starting up, then refresh the page."
            )
            st.stop()
        default_idx = 0
        for i, p in enumerate(players):
            if "Jokic" in str(p.get("player_name", "")):
                default_idx = i
                break
        player = st.selectbox(
            "Player (any active player with playoff history)",
            players,
            index=default_idx,
            format_func=lambda p: (
                f"{p.get('player_name', p.get('player_id'))} "
                f"(RS GS {p.get('rs_game_score_mean', 0):.1f})"
            ),
        )

        # All 30 teams
        teams = fetch_teams()
        opp_default = 0
        for i, t in enumerate(teams):
            if "Celtics" in str(t.get("team_name", "")):
                opp_default = i
                break
        opp_obj = st.selectbox(
            "Opponent (any team)",
            teams,
            index=opp_default,
            format_func=lambda t: t.get("team_name", str(t.get("team_id"))),
        )
        opponent_team_id = int(opp_obj.get("team_id", 0))
        opponent_name = opp_obj.get("team_name", "Opp")

        game_num = st.slider("Series game number", 1, 7, 1)

    is_home = st.radio("Location", ["Home", "Away"], index=0, horizontal=True) == "Home"

    # Series score going INTO this game
    score_options = []
    for w in range(min(game_num, 4)):
        for l in range(min(game_num, 4)):
            if w + l == game_num - 1:
                score_options.append(f"{w}-{l}")
    if not score_options:
        score_options = ["0-0"]
    series_score = st.selectbox(
        "Current series score going into this game (W-L)",
        score_options, index=0,
    )

    st.divider()
    st.header("2. Advanced (optional)")
    with st.expander("Override player season stats for what-if analysis"):
        st.caption(
            "Move sliders away from 0 to override the player's actual "
            "regular-season stats. Useful for hypothetical scenarios."
        )
        override_ppg = st.slider("Override regular-season PPG", 0.0, 40.0, 0.0)
        override_usage = st.slider("Override usage (proxy)", 0.0, 35.0, 0.0)
        override_fg = st.slider("Override FG%", 0.0, 0.7, 0.0, step=0.01)

    st.divider()
    predict_clicked = st.button(
        "🔮 Replay this game" if is_replay else "💡 Run prediction",
        type="primary",
        use_container_width=True,
    )


# ============ Main panel ============
if not predict_clicked:
    if is_replay:
        st.info(
            "👈 Pick a season, player, opponent, and game number in the "
            "sidebar — then click **Replay this game** to see the model's "
            "pre-game prediction next to the actual result."
        )
    else:
        st.info(
            "👈 Pick any player + opponent combination (including matchups "
            "that never actually happened) and click **Run prediction**. "
            "No actual result will be shown since this may be a hypothetical "
            "scenario."
        )

    st.markdown(
        """
        ### Two modes

        **🎬 Replay mode** — Pick an actual 2023–24 or 2024–25 playoff
        player-game. The model gives its pre-game prediction; the app then
        shows what really happened so you can see whether the prediction
        was right. *Restricted to completed seasons with full playoff data.*

        **💡 What-if mode** — Pick any player, any opponent, and any series
        situation. Use this to **predict 2025–26 playoff games that haven't
        happened yet** (using the player's 2025–26 regular-season stats as
        the baseline), or ask hypothetical questions like
        "how would Jokić do against the 1996 Bulls?"

        ### What "underperform" means

        Game Score > 20% below the player's regular-season average,
        with a 1.0 GS absolute floor.

        ### What the model uses

        45 features grouped into six families: player regular-season baseline
        (z-scored), player playoff history, current-game context, opponent
        strength, player-opponent matchup history, and in-series momentum.
        See the writeup for details.
        """
    )
else:
    overrides: dict[str, float] = {}
    if override_ppg > 0:
        overrides["pts_per_game"] = float(override_ppg)
    if override_usage > 0:
        overrides["usage_proxy_per_game"] = float(override_usage)
    if override_fg > 0:
        overrides["fg_pct"] = float(override_fg)

    payload = {
        "player_id": int(player.get("player_id", 0)),
        "season": season,
        "opponent_team_id": opponent_team_id,
        "series_game_number": int(game_num),
        "is_home": bool(is_home),
        "series_score": series_score,
        "feature_overrides": overrides,
    }

    with st.spinner("Running model..."):
        result = call_predict(payload)
        # Only fetch actual result in Replay mode
        actual = None
        if is_replay:
            actual = fetch_actual(
                int(player["player_id"]), season,
                opponent_team_id, int(game_num),
            )

    if result is None:
        st.stop()

    pred = result.get("prediction", {})
    ctx = result.get("context", {})
    expl = result.get("explanation", {})

    prob = float(pred.get("underperform_probability", 0.0))
    expected_gs = pred.get("expected_game_score")
    rs_avg = ctx.get("regular_season_gs_avg")

    if prob >= HIGH_RISK_THRESHOLD:
        risk_color = "#DC2626"
        risk_label = "High risk"
    elif prob <= LOW_RISK_THRESHOLD:
        risk_color = "#16A34A"
        risk_label = "Low risk"
    else:
        risk_color = "#D97706"
        risk_label = "Mixed signal"

    st.subheader(
        f"{ctx.get('player_name', 'Player')} vs. {opponent_name} "
        f"— Game {game_num} ({season})"
    )
    if not is_replay:
        st.caption(
            "💡 What-if mode: this may be a hypothetical matchup that never "
            "actually happened, so no real result is shown."
        )

    # ===== Section 1: Model prediction =====
    st.markdown("### 🤖 What the model predicts")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='color:#64748B;font-size:14px'>Underperform probability</div>"
            f"<div style='color:{risk_color};font-size:60px;font-weight:bold;'>"
            f"{prob * 100:.0f}%</div>"
            f"<div style='color:{risk_color}'>{risk_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(prob)

    with col2:
        if expected_gs is not None and rs_avg is not None:
            st.metric(
                "Predicted Game Score",
                f"{expected_gs:.1f}",
                delta=f"{(expected_gs - rs_avg):+.1f} vs RS avg",
                delta_color="inverse",
            )

    with col3:
        if rs_avg is not None:
            st.metric("Regular-season GS average", f"{rs_avg:.1f}")
        else:
            st.metric("Regular-season GS average", "n/a")

    # ===== Section 2: Actual result (Replay mode only) =====
    if is_replay and actual:
        st.divider()
        st.markdown("### 🏀 What actually happened")
        c1, c2, c3 = st.columns(3)

        actual_gs = actual.get("actual_game_score")
        actually_under = actual.get("actual_underperformed", False)
        rs_avg_actual = actual.get("regular_season_gs_avg", rs_avg)

        with c1:
            outcome_color = "#DC2626" if actually_under else "#16A34A"
            outcome_label = "UNDERPERFORMED" if actually_under else "DID NOT UNDERPERFORM"
            st.markdown(
                f"<div style='text-align:center'>"
                f"<div style='color:#64748B;font-size:14px'>Actual outcome</div>"
                f"<div style='color:{outcome_color};font-size:28px;font-weight:bold;'>"
                f"{outcome_label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with c2:
            if actual_gs is not None and rs_avg_actual is not None:
                st.metric(
                    "Actual Game Score",
                    f"{actual_gs:.1f}",
                    delta=f"{(actual_gs - rs_avg_actual):+.1f} vs RS avg",
                    delta_color="inverse",
                )

        with c3:
            model_says_under = prob >= 0.40
            correct = model_says_under == bool(actually_under)
            if correct:
                st.success("✅ Model called it correctly")
            else:
                st.error("❌ Model called it incorrectly")
    elif is_replay and actual is None:
        st.info(
            "Replay mode: but no actual record was found for this exact "
            "(player, opponent, game number) — the player may not have "
            "played that particular game."
        )

    # ===== Section 3: SHAP factors =====
    st.divider()
    st.markdown("### 🔍 Why this prediction? (SHAP top factors)")
    top = expl.get("top_factors", [])
    if not top:
        st.info("Explanations are not yet available.")
    for factor in top:
        v = factor.get("shap_value", 0.0)
        emoji = "🔴" if v > 0 else "🟢"
        st.markdown(
            f"{emoji} **{factor.get('human_readable', '')}** "
            f"_(contribution {v:+.2f})_"
        )

    with st.expander("Raw API response (debug)"):
        st.code(json.dumps({"predict": result, "actual": actual},
                           indent=2, default=str), language="json")
