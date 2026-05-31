"""
Tests for the relative underperformance threshold.

The threshold is max(rs_avg * 0.20, 1.0) — i.e. a player has to drop more
than 20% of their regular-season average, with a 1.0 GS floor.
"""
import pandas as pd
import pytest

from src.data.build_features import (
    underperform_threshold,
    RELATIVE_DROP,
    ABSOLUTE_FLOOR,
    MIN_RS_GAME_SCORE,
    MIN_RS_GAMES_PLAYED,
)


def test_constants():
    assert RELATIVE_DROP == 0.20
    assert ABSOLUTE_FLOOR == 1.0


def test_threshold_star_player():
    # LeBron RS_avg=25 -> max(25*0.2, 1.0) = max(5.0, 1.0) = 5.0
    assert underperform_threshold(pd.Series([25.0]))[0] == pytest.approx(5.0)


def test_threshold_role_player():
    # Role player RS_avg=8 -> max(8*0.2, 1.0) = max(1.6, 1.0) = 1.6
    assert underperform_threshold(pd.Series([8.0]))[0] == pytest.approx(1.6)


def test_threshold_rookie_floored():
    # Rookie RS_avg=3 -> max(3*0.2, 1.0) = max(0.6, 1.0) = 1.0 (floored)
    assert underperform_threshold(pd.Series([3.0]))[0] == pytest.approx(1.0)


def test_threshold_at_floor_boundary():
    # RS_avg=5 -> max(5*0.2, 1.0) = max(1.0, 1.0) = 1.0 (right at floor)
    assert underperform_threshold(pd.Series([5.0]))[0] == pytest.approx(1.0)


def test_threshold_vectorized():
    """Function must work on a Series of mixed inputs."""
    rs = pd.Series([25.0, 15.0, 8.0, 3.0, 0.0])
    th = underperform_threshold(rs)
    assert th[0] == pytest.approx(5.0)   # star
    assert th[1] == pytest.approx(3.0)   # solid starter
    assert th[2] == pytest.approx(1.6)   # role player
    assert th[3] == pytest.approx(1.0)   # rookie (floor kicks in)
    assert th[4] == pytest.approx(1.0)   # zero baseline (floor)


def test_threshold_scalar_input():
    """Function must also work on a plain Python float."""
    assert float(underperform_threshold(25.0)) == pytest.approx(5.0)


def test_fringe_filter_constants():
    """Verify the fringe-player filter thresholds are sane."""
    # MIN_RS_GAME_SCORE should be small but nonzero — fringe players excluded
    assert 1.0 < MIN_RS_GAME_SCORE < 10.0
    # MIN_RS_GAMES_PLAYED should require substantial playing time
    assert 10 <= MIN_RS_GAMES_PLAYED <= 40
