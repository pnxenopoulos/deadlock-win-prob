"""Tests for loading match data from deadlock-api."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest
import requests
from polars.testing import assert_frame_equal

from dwp.deadlock_api import load_latest_match_salts, load_recent_matches


def test_load_latest_match_salts(tmp_path: Path) -> None:
    frames = [pl.DataFrame({"match_id": [1]}), pl.DataFrame({"match_id": [2]})]
    for index, frame in enumerate(frames):
        frame.write_parquet(tmp_path / f"{index}.parquet")
    response = Mock()
    response.json.return_value = {
        "public_url": tmp_path.as_uri(),
        "tables": {
            "match_salts": {"files": [{"key": "0.parquet"}, {"key": "1.parquet"}]}
        },
    }

    with patch("dwp.deadlock_api.requests.get", return_value=response) as get:
        result = load_latest_match_salts()

    get.assert_called_once_with(
        "https://data.deadlock-api.com/v1/manifest.json", timeout=30
    )
    response.raise_for_status.assert_called_once_with()
    assert_frame_equal(result, pl.concat(frames))


def test_load_latest_match_salts_http_error() -> None:
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("503 Unavailable")

    with (
        patch("dwp.deadlock_api.requests.get", return_value=response),
        pytest.raises(requests.HTTPError, match="503"),
    ):
        load_latest_match_salts()

    response.json.assert_not_called()


@pytest.fixture
def match_player_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Mock:
    now = datetime(2026, 10, 22, tzinfo=UTC)
    clock = Mock(wraps=datetime)
    clock.now.return_value = now
    monkeypatch.setattr("dwp.deadlock_api.datetime", clock)
    frame = pl.DataFrame(
        {
            "match_id": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "match_mode": [
                "Ranked",
                "Ranked",
                "Unranked",
                "Ranked",
                "Ranked",
                "Ranked",
                "Ranked",
                None,
                "Ranked",
                "Ranked",
                "Ranked",
            ],
            "game_mode": [
                "Normal",
                "Normal",
                "Normal",
                "StreetBrawl",
                "Normal",
                "Normal",
                "Normal",
                "Normal",
                None,
                "Normal",
                "Normal",
            ],
            "match_outcome": [
                "TeamWin",
                "TeamWin",
                "TeamWin",
                "TeamWin",
                "Error",
                "Draw",
                None,
                "TeamWin",
                "TeamWin",
                "TeamWin",
                "TeamWin",
            ],
            "winning_team": [
                "Team0",
                "Team0",
                "Team1",
                "Team1",
                "Team0",
                "Team1",
                "Team0",
                "Team1",
                "Team0",
                "Team1",
                "Team1",
            ],
            "start_time": [datetime(2024, 1, 1, tzinfo=UTC)] * 11,
        },
        schema_overrides={"match_id": pl.UInt64},
    )
    frame.head(9).write_parquet(tmp_path / "recent.parquet")
    pl.concat([frame.head(1), frame.slice(9, 1)]).write_parquet(
        tmp_path / "boundary.parquet"
    )
    frame.tail(1).write_parquet(tmp_path / "three_days.parquet")
    response = Mock()
    response.json.return_value = {
        "public_url": tmp_path.as_uri(),
        "tables": {
            "match_player": {
                "files": [
                    {
                        "key": "recent.parquet",
                        "built_at": (now - timedelta(days=1)).isoformat(),
                    },
                    {
                        "key": "boundary.parquet",
                        "built_at": (now - timedelta(days=7)).isoformat(),
                    },
                    {
                        "key": "three_days.parquet",
                        "built_at": (now - timedelta(days=3)).isoformat(),
                    },
                    {
                        "key": "old_file_must_not_be_read.parquet",
                        "built_at": (
                            now - timedelta(days=7, microseconds=1)
                        ).isoformat(),
                    },
                ]
            }
        },
    }
    monkeypatch.setattr("dwp.deadlock_api.requests.get", Mock(return_value=response))
    return response


@pytest.mark.parametrize(
    ("days", "expected_ids", "expected_teams"),
    [
        (7, [1, 9, 10], ["Team0", "Team1", "Team1"]),
        (2, [1], ["Team0"]),
    ],
)
def test_load_recent_matches_defaults(
    match_player_response: Mock,
    days: int,
    expected_ids: list[int],
    expected_teams: list[str],
) -> None:
    result = load_recent_matches(days)

    expected = pl.DataFrame(
        {"match_id": expected_ids, "winning_team": expected_teams},
        schema={"match_id": pl.UInt64, "winning_team": pl.String},
    )
    assert_frame_equal(result, expected, check_row_order=False)
    match_player_response.raise_for_status.assert_called_once_with()


def test_load_recent_matches_mode_filters(match_player_response: Mock) -> None:
    result = load_recent_matches(
        7, match_modes=["Ranked", "Unranked"], game_modes=["Normal", "StreetBrawl"]
    )

    assert result["match_id"].sort().to_list() == [1, 2, 3, 9, 10]


def test_load_recent_matches_game_mode_filter(match_player_response: Mock) -> None:
    result = load_recent_matches(7, game_modes=["Normal"])

    assert result["match_id"].sort().to_list() == [1, 9, 10]


def test_load_recent_matches_empty_filters(match_player_response: Mock) -> None:
    assert load_recent_matches(7, match_modes=[]).is_empty()
    assert load_recent_matches(7, game_modes=[]).is_empty()
    assert load_recent_matches(7, match_modes=["Tutorial"]).is_empty()


def test_load_recent_matches_no_recent_files(match_player_response: Mock) -> None:
    match_player_response.json.return_value["tables"]["match_player"]["files"] = []

    result = load_recent_matches(7)

    assert result.is_empty()
    assert result.schema == {"match_id": pl.UInt64, "winning_team": pl.String}


@pytest.mark.parametrize("days", [0, -1])
def test_load_recent_matches_invalid_days(
    match_player_response: Mock, days: int
) -> None:
    with pytest.raises(ValueError, match="days must be positive"):
        load_recent_matches(days)

    match_player_response.raise_for_status.assert_not_called()


def test_load_recent_matches_http_error(match_player_response: Mock) -> None:
    match_player_response.raise_for_status.side_effect = requests.HTTPError(
        "503 Unavailable"
    )

    with pytest.raises(requests.HTTPError, match="503"):
        load_recent_matches(7)

    match_player_response.json.assert_not_called()


def test_limit_keeps_unique_matches_and_newer_files_first(
    match_player_response: Mock,
) -> None:
    result = load_recent_matches(7, limit=2)
    assert result["match_id"].to_list() == [1, 10]
    with pytest.raises(ValueError, match="limit"):
        load_recent_matches(7, limit=0)
