"""
NBA Playoff Underperformance Predictor — Flask API entry point.

Run locally:
    python api/app.py
    # → http://localhost:8080
    # → http://localhost:8080/docs  (Swagger UI)

Run in production (Cloud Run):
    gunicorn --bind :8080 --workers 2 --threads 4 app:app
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields
from marshmallow import ValidationError

# Make sure the project root is on sys.path so `import src.*` works inside the
# container (where the working dir is /app/api).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas.request_schemas import PredictRequestSchema  # noqa: E402
from api.services.feature_lookup import FeatureLookup, PlayerNotFoundError  # noqa: E402
from api.services.predictor import (  # noqa: E402
    Predictor, classify_confidence, classify_label, CLASSIFICATION_THRESHOLD,
)
from src.models.explain import ModelExplainer  # noqa: E402
from src.utils.logging_config import get_logger  # noqa: E402

LOG = get_logger("api")

ARTIFACTS_DIR = Path(os.environ.get(
    "ARTIFACTS_DIR", str(Path(__file__).parent / "artifacts")
))

# ---------- Build Flask app ----------
app = Flask(__name__)
CORS(app)
api = Api(
    app,
    version="1.0",
    title="NBA Playoff Underperformance Predictor",
    description=(
        "Predicts whether an NBA player will underperform their regular-season "
        "Game Score baseline in a specific playoff game."
    ),
    doc="/docs",
)
ns = api.namespace("v1", description="Predictions and lookups")


# ---------- Load model artifacts once at startup ----------
lookup = FeatureLookup(ARTIFACTS_DIR)
predictor = Predictor(ARTIFACTS_DIR / "model.pkl")
explainer: ModelExplainer | None = None  # built after artifacts load

try:
    lookup.load()
    LOG.info("Feature lookup loaded with %d feature columns.",
             len(lookup.feature_columns))
except Exception as e:
    LOG.warning("Feature lookup not yet populated: %s", e)

try:
    predictor.load()
    explainer = ModelExplainer(predictor.model, lookup.feature_columns)
    LOG.info("Model loaded from %s.", predictor.model_path)
except Exception as e:
    LOG.warning("Model not yet trained or missing: %s", e)


# ---------- OpenAPI models ----------
predict_input = api.model("PredictRequest", {
    "player_id": fields.Integer(required=True, example=2544,
                                description="NBA player ID"),
    "season": fields.String(required=True, example="2023-24",
                            description='Season string "YYYY-YY"'),
    "opponent_team_id": fields.Integer(required=True, example=1610612738),
    "series_game_number": fields.Integer(required=True, example=5,
                                         description="1–7"),
    "is_home": fields.Boolean(required=True, example=True),
    "series_score": fields.String(required=True, example="2-2"),
    "feature_overrides": fields.Raw(required=False, example={}),
})

predict_output = api.model("PredictResponse", {
    "prediction": fields.Raw,
    "context": fields.Raw,
    "explanation": fields.Raw,
})


# ---------- Routes ----------
@ns.route("/health")
class Health(Resource):
    def get(self):
        return {
            "status": "ok",
            "model_loaded": predictor._model is not None,
            "n_feature_columns": len(lookup.feature_columns),
        }


@ns.route("/players")
class Players(Resource):
    def get(self):
        """
        List players, optionally filtered to those who actually appeared in
        a given season's playoffs (set ?season=2024-25 to restrict).
        """
        from flask import request
        season = request.args.get("season")
        return {"players": lookup.list_players(limit=5000, playoff_season=season)}


@ns.route("/teams")
class Teams(Resource):
    def get(self):
        return {"teams": lookup.list_teams()}


@ns.route("/matchups")
class Matchups(Resource):
    def get(self):
        """
        For a given (player, season), return the list of opponents the player
        actually faced in the playoffs, and for each opponent how many games
        the series went. Used by the App to constrain dropdowns to plausible
        matchups.
        """
        from flask import request
        player_id = request.args.get("player_id", type=int)
        season = request.args.get("season", default="")
        if player_id is None or not season:
            api.abort(400, message="player_id and season are both required")
        return {"matchups": lookup.list_matchups(player_id, season)}


@ns.route("/actual_result")
class ActualResult(Resource):
    def get(self):
        """
        Return the actual Game Score the player put up in a specific playoff
        game, so the App can compare model prediction to reality.
        """
        from flask import request
        player_id = request.args.get("player_id", type=int)
        season = request.args.get("season", default="")
        opponent_team_id = request.args.get("opponent_team_id", type=int)
        series_game_number = request.args.get("series_game_number", type=int)
        if any(v is None for v in (player_id, opponent_team_id, series_game_number)) or not season:
            api.abort(400, message="player_id, season, opponent_team_id, series_game_number are all required")
        result = lookup.get_actual_result(
            player_id, season, opponent_team_id, series_game_number,
        )
        if result is None:
            api.abort(404, message="No record of this player playing that exact game.")
        return result


@ns.route("/predict")
class Predict(Resource):
    @ns.expect(predict_input)
    @ns.marshal_with(predict_output, code=200)
    def post(self):
        try:
            body = PredictRequestSchema().load(request.get_json(force=True) or {})
        except ValidationError as e:
            api.abort(400, message="Invalid input", details=e.messages)

        if predictor._model is None:
            api.abort(503, message="Model not yet loaded — server still starting "
                                   "or model artifact missing.")

        try:
            features, ctx = lookup.build_feature_vector(
                player_id=body["player_id"],
                season=body["season"],
                opponent_team_id=body["opponent_team_id"],
                series_game_number=body["series_game_number"],
                is_home=body["is_home"],
                series_score=body["series_score"],
                feature_overrides=body.get("feature_overrides") or {},
            )
        except PlayerNotFoundError as e:
            api.abort(404, message=str(e))

        prob = predictor.predict_proba(features)
        expected_gs = predictor.predict_expected_game_score(
            features,
            ctx.get("regular_season_gs_avg"),
            ctx.get("decline_threshold_gs"),
        )
        top_factors = (
            explainer.explain_one(features, top_k=3) if explainer else []
        )

        return {
            "prediction": {
                "underperform_probability": prob,
                "underperform_label": classify_label(prob),
                "classification_threshold": CLASSIFICATION_THRESHOLD,
                "confidence": classify_confidence(prob),
                "expected_game_score": expected_gs,
            },
            "context": ctx,
            "explanation": {"top_factors": top_factors},
        }


# ---------- Error handlers ----------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error_code": "NOT_FOUND", "message": str(e)}), 404


@app.errorhandler(500)
def server_error(e):
    LOG.exception("Internal server error")
    return jsonify({"error_code": "INTERNAL", "message": "Internal error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    LOG.info("Starting Flask dev server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
