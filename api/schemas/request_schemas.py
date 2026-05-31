"""Marshmallow schemas for API request validation."""
from marshmallow import Schema, fields, validate, ValidationError


VALID_SCORES = [f"{w}-{l}" for w in range(4) for l in range(4)
                if not (w == 3 and l == 3) or True]  # 0-0 through 3-3


class PredictRequestSchema(Schema):
    player_id = fields.Integer(required=True)
    season = fields.String(
        required=True,
        validate=validate.Regexp(r"^\d{4}-\d{2}$",
                                 error="season must look like 'YYYY-YY'"),
    )
    opponent_team_id = fields.Integer(required=True)
    series_game_number = fields.Integer(
        required=True, validate=validate.Range(min=1, max=7),
    )
    is_home = fields.Boolean(required=True)
    series_score = fields.String(
        required=True,
        validate=validate.Regexp(r"^[0-3]-[0-3]$",
                                 error="series_score must look like 'W-L', e.g. '2-1'"),
    )
    feature_overrides = fields.Dict(
        keys=fields.String(), values=fields.Float(),
        load_default=dict,
    )
