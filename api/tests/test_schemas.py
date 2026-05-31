"""Validate that the API request schema rejects bad input cleanly."""
import pytest
from marshmallow import ValidationError

from api.schemas.request_schemas import PredictRequestSchema


def _good_payload(**overrides):
    payload = {
        "player_id": 2544,
        "season": "2023-24",
        "opponent_team_id": 1610612738,
        "series_game_number": 5,
        "is_home": True,
        "series_score": "2-2",
    }
    payload.update(overrides)
    return payload


def test_valid_payload_passes():
    data = PredictRequestSchema().load(_good_payload())
    assert data["player_id"] == 2544
    assert data["season"] == "2023-24"
    assert data["is_home"] is True
    assert data["feature_overrides"] == {}


def test_missing_required_field():
    bad = _good_payload()
    del bad["player_id"]
    with pytest.raises(ValidationError):
        PredictRequestSchema().load(bad)


def test_bad_season_format():
    with pytest.raises(ValidationError):
        PredictRequestSchema().load(_good_payload(season="2023"))


def test_game_number_out_of_range():
    with pytest.raises(ValidationError):
        PredictRequestSchema().load(_good_payload(series_game_number=8))


def test_bad_series_score():
    with pytest.raises(ValidationError):
        PredictRequestSchema().load(_good_payload(series_score="5-0"))


def test_feature_overrides_passthrough():
    data = PredictRequestSchema().load(_good_payload(
        feature_overrides={"pts_per_game": 35.0, "fg_pct": 0.5}
    ))
    assert data["feature_overrides"] == {"pts_per_game": 35.0, "fg_pct": 0.5}
