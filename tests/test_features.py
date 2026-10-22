"""Training-only features, missingness, and causal graph tensor conversion."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import torch

from dwp.features import NUMERIC_FEATURES, GraphEncoder
from dwp.representations import GraphSnapshot


def test_training_normalization_and_missing_masks(
    graph_snapshot: GraphSnapshot,
) -> None:
    encoder = GraphEncoder.fit([graph_snapshot])
    means_before = [x.copy() for x in encoder.means]
    held_out = replace(
        graph_snapshot,
        players=graph_snapshot.players.with_columns(
            pl.Series("health", [None, 0], dtype=pl.Int64),
            pl.Series("hero_id", [999, None], dtype=pl.Int64),
            pl.Series("upgrade_ids", [[9999], None], dtype=pl.List(pl.Int64)),
        ),
    )
    encoded = encoder.transform(held_out)
    n = len(NUMERIC_FEATURES[1])
    health = NUMERIC_FEATURES[1].index("health")
    assert encoded.numeric[1][:, n + health].tolist() == [0, 1]
    assert encoded.numeric[1][0, health] == 0
    assert encoded.categorical[1][:, 0].tolist() == [1, 0]
    assert encoded.upgrades[:, 0].tolist() == [1, 0]
    for before, after in zip(means_before, encoder.means):
        np.testing.assert_array_equal(before, after)
    assert "999" not in encoder.vocabularies["hero_id"]
    assert "9999" not in encoder.vocabularies["upgrade_ids"]
    expected = np.mean(np.log1p([100, 50]))
    assert encoder.means[1][health] == pytest.approx(expected)
    for values in encoded.numeric:
        assert torch.isfinite(values).all()


def test_geometry_connectivity_and_known_empty_lists(
    graph_snapshot: GraphSnapshot,
) -> None:
    encoded = GraphEncoder.fit([graph_snapshot]).transform(graph_snapshot).to("cpu")
    assert encoded.node_types.tolist() == [0, 1, 1, 2, 2, 3, 3]
    assert encoded.teams.tolist() == [0, 3, 4, 3, 4, 1, 0]
    assert encoded.position_known.tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    torch.testing.assert_close(encoded.positions[2], torch.tensor([0.3, 0.4, 0]))
    assert encoded.connections.sum() == 1
    assert encoded.connections[5, 6]
    assert encoded.connections_known.tolist() == [False] * 5 + [True, False]
    item_count = NUMERIC_FEATURES[1].index("item_count")
    mask = len(NUMERIC_FEATURES[1]) + item_count
    assert encoded.numeric[1][:, mask].tolist() == [0, 1]


def test_ids_outcomes_and_unlisted_fields_are_ignored(
    graph_snapshot: GraphSnapshot,
) -> None:
    encoder = GraphEncoder.fit([graph_snapshot])
    changed = replace(
        graph_snapshot,
        tick=999999,
        match_id=321,
        build=111,
        players=graph_snapshot.players.with_columns(
            pl.lit(99999).alias("steam_id"),
            pl.lit("other").alias("node_id"),
            pl.lit(1).alias("winning_team"),
            pl.lit(100).alias("postgame_kills"),
        ),
        global_info=graph_snapshot.global_info.with_columns(pl.lit(0).alias("winner")),
    )
    a, b = encoder.transform(graph_snapshot), encoder.transform(changed)
    for x, y in zip(a.numeric + a.categorical, b.numeric + b.categorical):
        torch.testing.assert_close(x, y)
    assert encoder.vocabularies.keys() == {
        "hero_id",
        "objective_type",
        "phase",
        "lane",
        "upgrade_ids",
    }


def test_optional_baseline_stats(graph_snapshot: GraphSnapshot) -> None:
    enriched = replace(
        graph_snapshot,
        players=graph_snapshot.players.with_columns(
            pl.Series(
                "baseline_stats",
                [
                    [{"stat": "move_speed", "value": 7.0}],
                    [{"stat": "move_speed", "value": None}],
                ],
            )
        ),
    )
    encoded = GraphEncoder.fit([enriched]).transform(enriched)
    index = NUMERIC_FEATURES[1].index("baseline_move_speed")
    assert encoded.numeric[1][:, len(NUMERIC_FEATURES[1]) + index].tolist() == [1, 0]


def test_empty_entity_tables_and_rejected_empty_training(
    graph_snapshot: GraphSnapshot,
) -> None:
    empty = replace(
        graph_snapshot,
        players=graph_snapshot.players.clear(),
        objectives=graph_snapshot.objectives.clear(),
        ziplines=graph_snapshot.ziplines.clear(),
    )
    encoded = GraphEncoder.fit([empty]).transform(empty)
    assert [t.shape[0] for t in encoded.numeric] == [1, 0, 0, 0]
    assert encoded.upgrades.shape == (0, 1)
    assert encoded.connections.shape == (1, 1)
    with pytest.raises(ValueError, match="training graph"):
        GraphEncoder.fit([])
    with pytest.raises(ValueError, match="global node"):
        GraphEncoder.fit([replace(empty, global_info=empty.global_info.clear())])
