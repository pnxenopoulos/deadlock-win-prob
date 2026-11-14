"""Check team features and the versioned Kumo context."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest

from dwp.data.datasets import split_matches
from dwp.data.preparation import ReplayMatch
from dwp.graphs.snapshots import GraphSnapshot
from dwp.ml import tabular_training
from dwp.ml.tabular import FEATURE_COLUMNS, WINNER, feature_columns, tabular_features
from dwp.ml.tabular_features import ItemCatalog, LineupFeatures


@pytest.fixture
def tabular_graph(graph_snapshot: GraphSnapshot) -> GraphSnapshot:
    return replace(
        graph_snapshot,
        players=graph_snapshot.players.with_columns(
            spent_souls=pl.Series([200, 100]),
            gold_net_worth=pl.Series([700, 800]),
        ),
    )


def test_tabular_aggregates_player_state_by_side(tabular_graph: GraphSnapshot) -> None:
    features = tabular_features([tabular_graph])
    assert features.select(
        "hidden_king_souls",
        "archmother_souls",
        "archmother_health_fraction",
        "hidden_king_alive",
        "gold_net_worth_diff",
    ).row(0) == (500, 700, 0.5, 1, 100)


def test_tabular_keeps_unobserved_and_destroyed_objectives_distinct(
    tabular_graph: GraphSnapshot,
) -> None:
    graph = replace(
        tabular_graph,
        objectives=tabular_graph.objectives.with_columns(
            health=pl.when(pl.col("objective_type") == "walker")
            .then(0)
            .otherwise(pl.col("health"))
        ),
    )
    features = tabular_features([graph])
    assert features.select(
        "hidden_king_walker_standing",
        "hidden_king_walker_health_fraction",
        "archmother_walker_standing",
        "archmother_patron_phase",
    ).row(0) == (0, 0.0, None, 1)


@pytest.fixture
def item_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ItemCatalog:
    records = [
        {
            "ability_id": 1001,
            "definition": {
                "m_eAbilityType": "EAbilityType_Item",
                "m_eItemSlotType": "EItemSlotType_WeaponMod",
                "m_iItemTier": "EModTier_3",
            },
        },
        {
            "ability_id": 1002,
            "definition": {"m_eAbilityType": "EAbilityType_Signature"},
        },
    ]
    catalog = tmp_path / "abilities.json"
    catalog.write_text(json.dumps({"records": records}), encoding="utf-8")
    monkeypatch.setattr("dwp.ml.tabular_features.data.catalog_path", lambda *_: catalog)
    return ItemCatalog.load("1234")


def test_lineup_columns_use_training_vocabulary_and_current_team(
    tabular_graph: GraphSnapshot,
) -> None:
    lineups = LineupFeatures.fit([tabular_graph])
    swapped = replace(
        tabular_graph,
        players=tabular_graph.players.with_columns(
            hero_id=pl.Series([99, 11])
        ).reverse(),
    )
    states = tabular_features([swapped], lineups)
    assert lineups.hero_ids == (11, 22)
    assert states.select(
        "hidden_king_hero_11",
        "hidden_king_hero_22",
        "archmother_hero_11",
        "archmother_hero_22",
    ).row(0) == (0, 0, 1, 0)


@pytest.mark.parametrize(
    ("inventory", "expected"),
    [([1001, 1001, 1002], 1), ([1002], 0), ([], 0), (None, None), ([1001, 9999], None)],
)
def test_item_counts_distinguish_empty_and_unknown_inventories(
    tabular_graph: GraphSnapshot,
    item_catalog: ItemCatalog,
    inventory: list[int] | None,
    expected: int | None,
) -> None:
    graph = replace(
        tabular_graph,
        players=tabular_graph.players.with_columns(
            upgrade_ids=pl.Series([inventory, []], dtype=pl.List(pl.UInt32))
        ),
    )
    states = tabular_features([graph], LineupFeatures((11, 22), item_catalog))
    assert states.select(
        "hidden_king_item_weapon_tier_3_count", "archmother_item_weapon_tier_3_count"
    ).row(0) == (expected, 0)


@pytest.fixture
def model_bundle(
    tabular_graph: GraphSnapshot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    # A distinct clock value lets the test detect leaked validation/test rows.
    matches = [
        ReplayMatch(
            match_id,
            match_id % 2,
            "",
            tabular_graph.build,
            200,
            [
                replace(
                    tabular_graph,
                    match_id=match_id,
                    global_info=tabular_graph.global_info.with_columns(
                        elapsed_seconds=pl.lit(float(match_id))
                    ),
                )
            ],
        )
        for match_id in range(8)
    ]
    monkeypatch.setattr(tabular_training, "fit_kumo", Mock())
    monkeypatch.setattr(
        tabular_training,
        "predict_probabilities",
        lambda _, frame, **_kwargs: pl.DataFrame({"archmother": [0.5] * frame.height}),
    )
    monkeypatch.setattr(tabular_training, "package_version", lambda _: "0.2.1")
    output = tmp_path / "model"
    tabular_training.train_tabular(
        matches,
        output,
        model_version="model/kumo-win/v1.0.0",
        context_rows=3,
        features="baseline",
    )
    return output


def test_kumo_context_uses_only_training_matches(model_bundle: Path) -> None:
    training_ids = split_matches(pl.DataFrame({"match_id": range(8)})).train["match_id"]
    context = pl.read_parquet(model_bundle / "context.parquet")
    assert context.columns == [*FEATURE_COLUMNS, WINNER]
    assert context.height == 3
    assert set(context["elapsed_seconds"]) <= set(training_ids)
    assert set(context[WINNER]) == {"archmother", "hidden_king"}
    manifest = json.loads((model_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_version"] == "model/kumo-win/v1.0.0"


def test_kumo_loader_rejects_a_changed_context(model_bundle: Path) -> None:
    (model_bundle / "context.parquet").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum mismatch"):
        tabular_training.load_tabular_model(model_bundle)


def test_model_bundle_recreates_feature_order(
    model_bundle: Path,
    item_catalog: ItemCatalog,
) -> None:
    lineups = LineupFeatures((11, 22), item_catalog)
    path = model_bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        features=feature_columns(lineups),
        hero_ids=lineups.hero_ids,
        item_catalog={"version": item_catalog.version, "sha256": item_catalog.checksum},
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert tabular_training.load_tabular_features(model_bundle) == lineups


def test_model_bundle_reports_early_game_metrics(model_bundle: Path) -> None:
    metrics = json.loads((model_bundle / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_early"] == metrics["test"]
