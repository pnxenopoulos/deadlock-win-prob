"""Graph sampling, feature semantics, and causal state reconstruction."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest
from boon import Demo, HeroStat, StatMode
from polars.testing import assert_frame_equal

from dwp.representations import iter_graphs


@pytest.fixture
def demo(monkeypatch: pytest.MonkeyPatch) -> Mock:
    players = pl.DataFrame(
        {
            "tick": [0, 0, 10, 10, 10, 20, 20],
            "steam_id": [11, 22, 11, 22, None, 11, 22],
            "hero_id": [101, 102, 101, 102, 103, 101, 102],
            "x": [0, 3, 1, 4, 7, 2, 5],
            "y": [0, 4, 0, 4, 0, 0, 4],
            "z": [0] * 7,
            "health": [100, 80, 90, 70, 50, 0, 60],
            "max_health": [100] * 7,
            "barrier": [0, 5, 3, 0, 0, 0, 0],
            "level": [1, 1, 2, 2, 1, 3, 3],
            "souls": [0, 0, 50, 60, 0, 100, 110],
            "spent_souls": [0, 0, 10, 20, 0, 30, 40],
            "respawn_time": [0, 0, 0, 0, 0, 25, 0],
            "is_alive": [True, True, True, True, True, False, True],
            "upgrade_ids": [[1001, 9001], [], [1001], None, [1001], [1002], [1001]],
        },
        schema_overrides={
            "tick": pl.Int32,
            "steam_id": pl.UInt64,
            "upgrade_ids": pl.List(pl.UInt32),
        },
    )
    world = pl.DataFrame(
        {
            "tick": [20, 0, 10],
            "is_paused": [True, False, False],
            "next_midboss": [30.0, 50.0, 40.0],
        },
        schema_overrides={"tick": pl.Int32},
    )
    ziplines = pl.DataFrame(
        {
            "tick": [0, 0, 10, 20, 20],
            "node_handle": [100, 200, 100, 100, 300],
            "lane": [1] * 5,
            "x": [2, 3, 2, 2, 4],
            "y": [0] * 5,
            "z": [0] * 5,
            "connections": [[200], [100], None, [], [100]],
            "capturable": [True] * 5,
            "team_num": [2, 3, 3, None, 0],
        },
        schema_overrides={
            "tick": pl.Int32,
            "node_handle": pl.UInt32,
            "connections": pl.List(pl.UInt32),
        },
    )
    parsed = Mock(spec=Demo)
    parsed.match_id = 123
    parsed.build = 999
    parsed.snapshots.return_value = {
        "player_ticks": players,
        "world_ticks": world,
        "ziplines": ziplines,
    }
    parsed.players = pl.DataFrame(
        {"steam_id": [11, 22], "team_num": [2, 3]},
        schema_overrides={"steam_id": pl.UInt64},
    )
    parsed.objectives = pl.DataFrame(
        {
            "tick": [0, 12, 18, 30, 5, 15],
            "entity_id": [5, 5, 5, 9, 8, 8],
            "entity_serial": [1, 1, 2, 1, 2, 2],
            "objective_type": [
                "walker",
                "walker",
                "walker",
                "patron",
                "mid_boss",
                "mid_boss",
            ],
            "team_num": [2, 2, 3, 3, 0, 0],
            "health": [100, 0, 200, 999, 1000, 0],
            "max_health": [100, 100, 200, 999, 1000, 1000],
            "phase": [0] * 6,
            "lane": [1, 1, 1, 0, 0, 0],
            "x": [0, 0, 0, 9, 2, 2],
            "y": [5, 5, 6, 9, 2, 2],
            "z": [0] * 6,
        },
        schema_overrides={
            "tick": pl.Int32,
            "entity_id": pl.Int32,
            "entity_serial": pl.UInt32,
        },
    )
    parsed.mid_boss = pl.DataFrame(
        {
            "tick": [5, 15, 25],
            "event": ["spawned", "killed", "spawned"],
            "team_num": [0, 2, 0],
        },
        schema_overrides={"tick": pl.Int32},
    )
    parsed.tick_to_seconds.side_effect = {0: 0.0, 10: 1.0, 20: 1.5}.__getitem__
    parsed.tick_to_match_seconds.side_effect = {0: -0.5, 10: 0.5, 20: 1.0}.__getitem__
    monkeypatch.setattr(
        "dwp.representations.data.catalog_path",
        Mock(side_effect=AssertionError("unexpected catalog access")),
    )
    return parsed


@pytest.fixture
def catalog(demo: Mock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Mock:
    path = tmp_path / "abilities.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "ability_id": 1001,
                        "definition": {"m_eAbilityType": "EAbilityType_Item"},
                    },
                    {
                        "ability_id": 1002,
                        "definition": {"m_eAbilityType": "EAbilityType_Item"},
                    },
                    {
                        "ability_id": 9001,
                        "definition": {"m_eAbilityType": "EAbilityType_Signature"},
                    },
                ]
            }
        )
    )
    getter = Mock(return_value=path)
    monkeypatch.setattr("dwp.representations.data.catalog_path", getter)
    demo.calculate_hero_stats.return_value = SimpleNamespace(
        values=pl.DataFrame(
            {
                "tick": [0, 10],
                "steam_id": [11, 22],
                "hero_id": [101, 102],
                "stat": ["move_speed", "move_speed"],
                "value": [7.0, None],
                "unit": ["m/s", "m/s"],
                "mode": ["baseline", "baseline"],
                "ruleset": ["v1", "v1"],
                "status": ["calculated", "unresolved"],
                "diagnostic": [None, "missing source"],
            },
            schema_overrides={"tick": pl.Int32, "steam_id": pl.UInt64},
        )
    )
    return getter


def test_sampling_and_recorded_features(demo: Mock) -> None:
    original = demo.snapshots.return_value["player_ticks"].clone()
    graphs = list(iter_graphs(demo, every=10))

    assert [graph.tick for graph in graphs] == [0, 10, 20]
    demo.snapshots.assert_called_once_with(
        ["player_ticks", "world_ticks", "ziplines"],
        every=10,
        ticks=None,
        start_tick=None,
        end_tick=None,
    )
    for graph in graphs:
        assert (
            graph.match_id == 123 and graph.build == 999 and graph.data_version is None
        )
        assert graph.nodes["node_id"].n_unique() == graph.nodes.height
        assert graph.players["item_ids"].null_count() == graph.players.height
        assert graph.players["baseline_stats"].null_count() == graph.players.height
        assert "winning_team" not in graph.global_info.columns
    assert [graph.global_info["elapsed_seconds"][0] for graph in graphs] == [
        0.0,
        1.0,
        1.5,
    ]
    assert [graph.global_info["match_time_seconds"][0] for graph in graphs] == [
        -0.5,
        0.5,
        1.0,
    ]
    assert graphs[0].players["upgrade_ids"][0].to_list() == [1001, 9001]
    assert graphs[-1].players.filter(pl.col("steam_id") == 11)["respawn_time"][0] == 25
    assert graphs[1].players.filter(pl.col("steam_id") == 11)["barrier"][0] == 3
    demo.calculate_hero_stats.assert_not_called()
    assert_frame_equal(demo.snapshots.return_value["player_ticks"], original)


def test_objective_state_is_causal_and_keeps_lifetimes(demo: Mock) -> None:
    graphs = list(iter_graphs(demo, every=10))

    assert graphs[0].objectives["node_id"].to_list() == ["objective:5:1"]
    assert (
        graphs[1].objectives.filter(pl.col("node_id") == "objective:5:1")["health"][0]
        == 100
    )
    last = graphs[-1].objectives
    assert last.filter(pl.col("node_id") == "objective:5:1")["destroyed"][0]
    assert last.filter(pl.col("node_id") == "objective:5:2")["health"][0] == 200
    for graph in graphs:
        assert "objective:9:1" not in graph.objectives["node_id"].to_list()
        assert (graph.objectives["observed_tick"] <= graph.tick).all()
    assert [graph.global_info["mid_boss_up"][0] for graph in graphs] == [
        None,
        True,
        False,
    ]


def test_zipline_snapshot_control_and_missing_geometry(demo: Mock) -> None:
    first, middle, last = list(iter_graphs(demo, every=10))

    assert "zipline:200" in first.ziplines["node_id"].to_list()
    assert "zipline:200" not in middle.ziplines["node_id"].to_list()
    assert "zipline:300" not in middle.ziplines["node_id"].to_list()
    assert "zipline:300" in last.ziplines["node_id"].to_list()
    assert middle.ziplines["team_num"][0] == 3
    assert middle.ziplines["connections"][0] is None
    assert (
        last.ziplines.filter(pl.col("node_handle") == 100)["connections"][0].to_list()
        == []
    )


def test_complete_typed_edges_and_distances(demo: Mock) -> None:
    graph = next(iter_graphs(demo, every=10))
    edges = graph.edges

    assert edges.height == graph.nodes.height * (graph.nodes.height - 1)
    assert not (edges["source"] == edges["target"]).any()
    enemy = edges.filter(
        (pl.col("source") == "player:11") & (pl.col("target") == "player:22")
    ).row(0, named=True)
    assert enemy["source_type"] == enemy["target_type"] == "player"
    assert enemy["source_team"] == 2 and enemy["target_team"] == 3
    assert enemy["team_relation"] == "opposing_team" and enemy["distance"] == 5.0
    friendly = edges.filter(
        (pl.col("source") == "player:11") & (pl.col("target") == "zipline:100")
    )
    assert friendly["team_relation"][0] == "same_team"
    global_edges = edges.filter(
        (pl.col("source_type") == "global") | (pl.col("target_type") == "global")
    )
    assert set(global_edges["team_relation"].to_list()) == {"global"}
    assert global_edges["distance"].null_count() == global_edges.height
    last = list(iter_graphs(demo, every=10))[-1].edges.filter(
        pl.col("source") == "player:11"
    )
    assert (
        last.filter(pl.col("target") == "zipline:100")["team_relation"][0] == "unknown"
    )
    assert (
        last.filter(pl.col("target") == "zipline:300")["team_relation"][0] == "neutral"
    )


def test_optional_catalog_and_baseline_enrichment(demo: Mock, catalog: Mock) -> None:
    first, middle, _ = list(
        iter_graphs(
            demo, every=10, data_version="test", baseline_stats=[HeroStat.MOVE_SPEED]
        )
    )

    catalog.assert_called_once_with("abilities", "test")
    kwargs = demo.calculate_hero_stats.call_args.kwargs
    assert kwargs["ticks"] == [0, 10, 20] and set(kwargs["steam_ids"]) == {11, 22}
    assert kwargs["mode"] == StatMode.BASELINE and kwargs["strict"] is False
    assert kwargs["data_version"] == "test" and kwargs["stats"] == [HeroStat.MOVE_SPEED]
    player = first.players.filter(pl.col("steam_id") == 11)
    assert player["item_ids"][0].to_list() == [1001]
    assert player["upgrade_ids"][0].to_list() == [1001, 9001]
    assert player["baseline_stats"][0].to_list()[0]["value"] == 7.0
    unresolved = middle.players.filter(pl.col("steam_id") == 22)["baseline_stats"][
        0
    ].to_list()[0]
    assert unresolved["value"] is None and unresolved["diagnostic"] == "missing source"
    assert unresolved["status"] == "unresolved"
    assert (
        middle.players.filter(pl.col("steam_id").is_null())["baseline_stats"][0] is None
    )
    assert first.data_version == "test"


def test_catalog_can_classify_items_without_calculating_stats(
    demo: Mock, catalog: Mock
) -> None:
    graph = next(iter_graphs(demo, every=10, data_version="test", baseline_stats=()))

    assert graph.players["item_ids"][0].to_list() == [1001]
    demo.calculate_hero_stats.assert_not_called()


def test_ambiguous_player_identity_is_not_merged(demo: Mock, catalog: Mock) -> None:
    frame = demo.snapshots.return_value["player_ticks"]
    demo.snapshots.return_value["player_ticks"] = pl.concat([frame, frame.head(1)])
    graph = next(iter_graphs(demo, every=10, data_version="test"))
    ambiguous = graph.players.filter(pl.col("steam_id") == 11)

    assert ambiguous.height == 2
    assert ambiguous["node_id"].n_unique() == 2
    assert not ambiguous["identity_known"].any()
    assert ambiguous["baseline_stats"].null_count() == 2


def test_ambiguous_roster_team_stays_unknown(demo: Mock) -> None:
    demo.players = pl.concat(
        [
            demo.players,
            demo.players.head(1).with_columns(team_num=pl.lit(3, dtype=pl.Int64)),
        ]
    )
    graph = next(iter_graphs(demo, every=10))

    assert graph.players.filter(pl.col("steam_id") == 11)["team_num"][0] is None


def test_missing_match_clock_is_not_guessed(demo: Mock) -> None:
    demo.tick_to_match_seconds.side_effect = None
    demo.tick_to_match_seconds.return_value = None
    graph = next(iter_graphs(demo, every=10))

    assert graph.global_info["match_time_seconds"][0] is None
    assert graph.global_info["elapsed_seconds"][0] == 0.0


def test_empty_entity_histories(demo: Mock) -> None:
    demo.objectives = demo.objectives.clear()
    demo.mid_boss = demo.mid_boss.clear()
    demo.snapshots.return_value["ziplines"] = demo.snapshots.return_value[
        "ziplines"
    ].clear()
    graph = next(iter_graphs(demo, every=10))

    assert graph.objectives.is_empty() and graph.ziplines.is_empty()
    assert graph.global_info["mid_boss_up"][0] is None
    assert graph.edges.height == graph.nodes.height * (graph.nodes.height - 1)


def test_empty_snapshots(demo: Mock) -> None:
    demo.snapshots.return_value = {
        name: frame.clear() for name, frame in demo.snapshots.return_value.items()
    }

    assert list(iter_graphs(demo, every=10)) == []
    demo.calculate_hero_stats.assert_not_called()


def test_tick_window_is_forwarded(demo: Mock) -> None:
    demo.snapshots.return_value = {
        name: frame.filter(pl.col("tick") == 10)
        for name, frame in demo.snapshots.return_value.items()
    }
    graphs = list(iter_graphs(demo, every=10, start_tick=10, end_tick=10))

    assert [graph.tick for graph in graphs] == [10]
    assert (
        demo.snapshots.call_args.kwargs["start_tick"]
        == demo.snapshots.call_args.kwargs["end_tick"]
        == 10
    )


@pytest.mark.parametrize("every", [0, -1])
def test_invalid_interval(demo: Mock, every: int) -> None:
    with pytest.raises(ValueError, match="every must be positive"):
        list(iter_graphs(demo, every=every))

    demo.snapshots.assert_not_called()


def test_invalid_catalog_version(demo: Mock) -> None:
    with pytest.raises(ValueError, match="nonempty version"):
        list(iter_graphs(demo, data_version=""))


def test_explicit_ticks_override_stride(demo: Mock) -> None:
    list(iter_graphs(demo, ticks=[0, 20], end_tick=20))
    demo.snapshots.assert_called_once_with(
        ["player_ticks", "world_ticks", "ziplines"],
        every=None,
        ticks=[0, 20],
        start_tick=None,
        end_tick=20,
    )
