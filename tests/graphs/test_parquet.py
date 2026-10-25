"""Nested/nullable Parquet round trips, empty workers, and strict merging."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from random import Random
from unittest.mock import Mock

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dwp.cli.train import main
from dwp.data.planning import PreparationPlan, plan_matches, write_plan
from dwp.data.preparation import ReplayMatch
from dwp.graphs.parquet import load_graph_shards, save_graph_shard
from dwp.graphs.snapshots import GraphSnapshot


@pytest.fixture
def graph_dataset(
    graph_snapshot: GraphSnapshot, monkeypatch: pytest.MonkeyPatch
) -> tuple[PreparationPlan, list[list[ReplayMatch]]]:
    source = pl.DataFrame(
        {
            "match_id": [100, 101, 102, 103],
            "replay_url": [f"https://example.com/{i}.dem.bz2" for i in range(100, 104)],
        }
    )
    monkeypatch.setattr(
        "dwp.data.planning.select_candidates", Mock(return_value=source)
    )
    plan = plan_matches(matches=4, workers=2, max_snapshots=2)
    # Preserve nullable lists/structs, UInt64 identities, and known-empty inventories.
    baseline = pl.Series(
        "baseline_stats",
        [
            [{"stat": "move_speed", "value": 7.0}],
            None,
        ],
        dtype=pl.List(pl.Struct({"stat": pl.String, "value": pl.Float64})),
    )
    graph = replace(
        graph_snapshot,
        players=graph_snapshot.players.with_columns(
            baseline,
            pl.col("steam_id").cast(pl.UInt64),
        ),
    )
    groups = []
    for shard in plan["shards"]:
        group = []
        for candidate in shard["replays"]:
            mid = candidate["match_id"]
            first = replace(graph, match_id=mid, tick=10)
            second = replace(
                first,
                tick=20,
                objectives=first.objectives.clear(),
                ziplines=first.ziplines.with_columns(
                    pl.Series("connections", [[], None], dtype=pl.List(pl.Int64))
                ),
            )
            group.append(
                ReplayMatch(
                    mid,
                    mid % 2,
                    candidate["replay_url"],
                    graph.build,
                    1000,
                    [first, second],
                )
            )
        groups.append(group)
    return plan, groups


def test_parquet_round_trip_preserves_all_graph_features(
    graph_dataset, tmp_path: Path
) -> None:
    plan, groups = graph_dataset
    for i, matches in enumerate(groups):
        save_graph_shard(matches, tmp_path / f"shard-{i}", plan, i)
        assert len(list((tmp_path / f"shard-{i}").glob("*.parquet"))) == 5
    loaded = load_graph_shards(tmp_path, plan)
    original = sorted([m for group in groups for m in group], key=lambda m: m.match_id)
    for actual, expected in zip(loaded, original, strict=True):
        assert (
            actual.match_id,
            actual.label,
            actual.replay_url,
            actual.build,
            actual.game_over_tick,
        ) == (
            expected.match_id,
            expected.label,
            expected.replay_url,
            expected.build,
            expected.game_over_tick,
        )
        for a, b in zip(actual.graphs, expected.graphs, strict=True):
            assert a.tick == b.tick and a.data_version == b.data_version
            for table in ("global_info", "players", "objectives", "ziplines"):
                assert_frame_equal(getattr(a, table), getattr(b, table))
    assert original[0].graphs[0].players.columns == loaded[0].graphs[0].players.columns


def test_empty_worker_is_allowed_but_missing_worker_is_not(
    graph_dataset, tmp_path: Path
) -> None:
    plan, groups = graph_dataset
    save_graph_shard(groups[0], tmp_path / "shard-0", plan, 0)
    with pytest.raises(ValueError, match="missing"):
        load_graph_shards(tmp_path, plan)
    save_graph_shard([], tmp_path / "shard-1", plan, 1)
    assert [m.match_id for m in load_graph_shards(tmp_path, plan)] == [100, 102]


def test_mixed_or_duplicate_shards_are_rejected(graph_dataset, tmp_path: Path) -> None:
    plan, groups = graph_dataset
    for i, matches in enumerate(groups):
        save_graph_shard(matches, tmp_path / f"shard-{i}", plan, i)
    shutil.copytree(tmp_path / "shard-0", tmp_path / "duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        load_graph_shards(tmp_path, plan)
    metadata = tmp_path / "duplicate" / "dataset.json"
    content = json.loads(metadata.read_text())
    content["plan_id"] = "another-plan"
    metadata.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="mixed"):
        load_graph_shards(tmp_path, plan)


def test_assignment_and_future_state_are_rejected(
    graph_dataset, tmp_path: Path
) -> None:
    plan, groups = graph_dataset
    with pytest.raises(ValueError, match="assigned"):
        save_graph_shard(groups[1], tmp_path / "wrong", plan, 0)
    invalid = replace(groups[0][0], game_over_tick=20)
    with pytest.raises(ValueError, match="post-game"):
        save_graph_shard([invalid], tmp_path / "future", plan, 0)


def test_final_training_command_reads_parquet_without_network(
    graph_dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, groups = graph_dataset
    for i, matches in enumerate(groups):
        save_graph_shard(matches, tmp_path / "prepared" / f"shard-{i}", plan, i)
    plan_path = tmp_path / "plan.json"
    write_plan(plan, plan_path)
    source = Mock(
        side_effect=AssertionError("offline training must not rediscover/download")
    )
    monkeypatch.setattr("dwp.cli.train.prepare_matches", source)
    output = tmp_path / "model"
    main(
        [
            "--prepared-data",
            str(tmp_path / "prepared"),
            "--plan",
            str(plan_path),
            "--epochs",
            "1",
            "--hidden-dim",
            "16",
            "--num-layers",
            "1",
            "--batch-size",
            "1",
            "--threads",
            "1",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )
    source.assert_not_called()
    assert (output / "model.pt").is_file() and (
        output / "test_calibration.png"
    ).is_file()
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["data"]["preparation_plan"]["plan_id"] == plan["plan_id"]
    matches = pl.read_csv(output / "matches.csv")
    assert matches["match_id"].n_unique() == 4
    assert {"train", "validation", "test"} == set(matches["split"])


def test_preparation_commands_plan_once_and_allow_single_match_workers(
    graph_dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dwp.cli.prepare import main as prepare_main

    plan, groups = graph_dataset
    plan_path = tmp_path / "plan.json"
    prepare_main(
        [
            "plan",
            "--matches",
            "4",
            "--workers",
            "2",
            "--max-parallel",
            "1",
            "--max-snapshots",
            "2",
            "--output",
            str(plan_path),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["matrix"] == {"shard": [0, 1]} and result["max_parallel"] == 1
    assert result["plan_id"] == plan["plan_id"]
    parse = Mock(return_value=groups[0][:1])
    monkeypatch.setattr("dwp.cli.prepare.prepare_replays", parse)
    output = tmp_path / "worker"
    prepare_main(
        ["shard", "--plan", str(plan_path), "--shard-id", "0", "--output", str(output)]
    )
    assert parse.call_args.kwargs["matches"] == 2
    assert pl.read_parquet(output / "matches.parquet").height == 1
    assert json.loads(capsys.readouterr().out)["matches"] == 1
    assert json.loads((output / "dataset.json").read_text())["shard_id"] == 0


def test_loader_rejects_missing_feature_files_and_duplicate_matches(
    graph_dataset, tmp_path: Path
) -> None:
    plan, groups = graph_dataset
    for i, matches in enumerate(groups):
        save_graph_shard(matches, tmp_path / f"shard-{i}", plan, i)
    second = tmp_path / "shard-1"
    # Corrupt the second worker to contain an already-seen match, including its table keys.
    metadata = pl.read_parquet(second / "matches.parquet").with_columns(
        pl.when(pl.col("match_id") == 101)
        .then(pl.lit(100, dtype=pl.UInt64))
        .otherwise(pl.col("match_id"))
        .alias("match_id")
    )
    metadata.write_parquet(second / "matches.parquet")
    for name in ("global_info", "players", "objectives", "ziplines"):
        path = second / f"{name}.parquet"
        table = pl.read_parquet(path).with_columns(
            pl.when(pl.col("_match_id") == 101)
            .then(pl.lit(100, dtype=pl.UInt64))
            .otherwise(pl.col("_match_id"))
            .alias("_match_id")
        )
        table.write_parquet(path)
    with pytest.raises(ValueError, match="duplicate match IDs"):
        load_graph_shards(tmp_path, plan)
    (tmp_path / "shard-0" / "players.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        load_graph_shards(tmp_path, plan)


def test_snapshots_are_lazy_and_read_only_requested_ranges(
    graph_dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, groups = graph_dataset
    for i, matches in enumerate(groups):
        save_graph_shard(matches, tmp_path / f"shard-{i}", plan, i)
    snapshot = Mock(wraps=GraphSnapshot)
    monkeypatch.setattr("dwp.graphs.parquet.GraphSnapshot", snapshot)
    loaded = load_graph_shards(tmp_path, plan)
    # Indexing thousands of matches must not create any per-snapshot DataFrames.
    assert loaded[0].snapshot_keys == ((100, 10), (100, 20))
    snapshot.assert_not_called()
    assert all(not isinstance(m.graphs, list) for m in loaded)
    graphs = loaded[0].graphs
    assert len(graphs) == 2
    first = graphs[0]
    assert first.tick == 10 and graphs[-1].tick == 20
    assert_frame_equal(first.players, groups[0][0].graphs[0].players)
    assert [g.tick for g in graphs[:1]] == [10]
    assert Random(42).choice(graphs).tick in (10, 20)
    with pytest.raises(IndexError):
        _ = graphs[2]
    # Access does not leave GraphSnapshot objects retained by the sequence.
    assert graphs[0] is not first


def test_loader_rejects_unlisted_rows_and_noncontiguous_snapshots(
    graph_dataset, tmp_path: Path
) -> None:
    plan, groups = graph_dataset
    for i, matches in enumerate(groups):
        save_graph_shard(matches, tmp_path / f"shard-{i}", plan, i)
    path = tmp_path / "shard-0" / "players.parquet"
    original = pl.read_parquet(path)
    pl.concat([original, original.head(1)]).write_parquet(path)
    with pytest.raises(ValueError, match="contiguous"):
        load_graph_shards(tmp_path, plan)
    original.with_columns(pl.lit(999999).alias("_match_id")).write_parquet(path)
    with pytest.raises(ValueError, match="unlisted"):
        load_graph_shards(tmp_path, plan)


def test_temporary_graph_storage_lives_with_returned_matches(
    graph_dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tempfile import TemporaryDirectory

    plan, groups = graph_dataset
    for i, matches in enumerate(groups):
        save_graph_shard(matches, tmp_path / f"shard-{i}", plan, i)
    workspaces = []

    def workspace(**kwargs):
        temporary = TemporaryDirectory(dir=tmp_path, **kwargs)
        workspaces.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr("dwp.graphs.parquet.TemporaryDirectory", workspace)
    loaded = load_graph_shards(tmp_path, plan)
    assert len(workspaces) == 1 and workspaces[0].is_dir()
    assert list(workspaces[0].glob("*.arrow"))
    assert loaded[0].graphs[0].tick == 10
    del loaded
    assert not workspaces[0].exists()
