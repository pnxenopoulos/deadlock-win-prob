"""Replay selection, replacement, label provenance, and pre-outcome sampling."""

from dataclasses import replace
from unittest.mock import Mock

import polars as pl
import pytest
import requests
from boon import Demo

from dwp.data.preparation import prepare_matches
from dwp.graphs.snapshots import GraphSnapshot


@pytest.fixture
def source(
    monkeypatch: pytest.MonkeyPatch, graph_snapshot: GraphSnapshot
) -> dict[str, Mock]:
    ids = list(range(100, 108))
    recent = Mock(
        return_value=pl.DataFrame(
            {"match_id": ids, "winning_team": ["Team0"] * len(ids)}
        )
    )
    salts = Mock(
        return_value=pl.DataFrame(
            {"match_id": ids + [9999], "cluster_id": [1] * 9, "replay_salt": [7] * 9}
        )
    )

    def replay(url: str, **kwargs) -> Mock:
        match_id = int(url.rsplit("/", 1)[-1].split("_")[0])
        demo = Mock(spec=Demo)
        demo.match_id = match_id
        demo.build = 999
        demo.game_over_tick = 200
        demo.winning_team_num = 2 + match_id % 2
        return demo

    def graphs(demo: Mock, **kwargs):
        return iter(
            replace(graph_snapshot, match_id=demo.match_id, tick=t)
            for t in [1, 20, 40, 60, 80, 200, 220]
        )

    loaders = {
        "load_recent_matches": recent,
        "load_latest_match_salts": salts,
        "load_replay": Mock(side_effect=replay),
        "iter_graphs": Mock(side_effect=graphs),
    }
    for name, function in loaders.items():
        monkeypatch.setattr(f"dwp.data.preparation.{name}", function)
    return loaders


def test_sampling_is_reproducible_and_uses_native_winner(
    source: dict[str, Mock],
) -> None:
    a = prepare_matches(matches=3, max_snapshots=2, seed=4, data_version="v1")
    b = prepare_matches(matches=3, max_snapshots=2, seed=4, data_version="v1")
    assert [(m.match_id, [g.tick for g in m.graphs]) for m in a] == [
        (m.match_id, [g.tick for g in m.graphs]) for m in b
    ]
    assert len(a) == 3 and len({m.match_id for m in a}) == 3
    for match in a:
        assert match.label == match.match_id % 2
        assert len(match.graphs) == 2 and all(g.tick < 200 for g in match.graphs)
    source["load_recent_matches"].assert_called_with(
        1, ("Ranked",), ("Normal",), limit=100
    )
    assert source["iter_graphs"].call_args.kwargs == {
        "every": 4096,
        "end_tick": 199,
        "data_version": "v1",
    }
    assert source["load_replay"].call_args.kwargs["session"] is not None


def test_failed_candidates_are_replaced(
    source: dict[str, Mock], caplog: pytest.LogCaptureFixture
) -> None:
    original = source["load_replay"].side_effect
    called = 0

    def flaky(url: str, **kwargs) -> Mock:
        nonlocal called
        called += 1
        if called == 1:
            raise requests.HTTPError("404 expired")
        demo = original(url, **kwargs)
        if called == 2:
            demo.winning_team_num = None
        if called == 3:
            demo.match_id = 99999
        return demo

    source["load_replay"].side_effect = flaky
    matches = prepare_matches(matches=3)
    assert len(matches) == 3 and called == 6
    assert (
        "404 expired" in caplog.text
        and "missing game-over" in caplog.text
        and "different match ID" in caplog.text
    )


def test_too_few_usable_replays_fails(source: dict[str, Mock]) -> None:
    source["load_replay"].side_effect = requests.HTTPError("expired")
    with pytest.raises(ValueError, match="at least three"):
        prepare_matches(matches=3)


def test_no_eligible_outcomes_avoids_salts_download(source: dict[str, Mock]) -> None:
    source["load_recent_matches"].return_value = pl.DataFrame(
        {"match_id": [1, 2, 3], "winning_team": ["Spectator"] * 3}
    )
    with pytest.raises(ValueError, match="eligible"):
        prepare_matches()
    source["load_latest_match_salts"].assert_not_called()


@pytest.mark.parametrize(
    "settings",
    [
        {"days": 0},
        {"matches": 2},
        {"every": 0},
        {"max_snapshots": 0},
        {"matches": 4, "candidate_limit": 3},
    ],
)
def test_invalid_options_do_not_fetch(source: dict[str, Mock], settings: dict) -> None:
    with pytest.raises(ValueError):
        prepare_matches(**settings)
    source["load_recent_matches"].assert_not_called()
