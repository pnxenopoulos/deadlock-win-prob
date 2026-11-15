"""Evaluate and package a versioned KumoTabular context from prepared matches."""

import json
import re
from collections.abc import Sequence
from hashlib import file_digest
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from dwp.data.datasets import split_matches
from dwp.data.preparation import ReplayMatch
from dwp.ml.metrics import ProbabilityMetrics, probability_metrics
from dwp.ml.tabular import (
    WINNER,
    feature_columns,
    fit_kumo,
    predict_probabilities,
    tabular_features,
)
from dwp.ml.tabular_features import (
    DEFAULT_FEATURE_SET,
    FeatureSet,
    ItemCatalog,
    LineupFeatures,
)

if TYPE_CHECKING:
    from sdm.models import KumoTabular

CONTEXT_FILE = "context.parquet"
EARLY_GAME_SECONDS = 600
DEFAULT_CONTEXT_ROWS = 2048
DEFAULT_NUM_ESTIMATORS = 8


def _features(matches: Sequence[ReplayMatch], lineups: LineupFeatures) -> pl.DataFrame:
    return pl.concat(
        tabular_features(match.graphs, lineups).with_columns(
            label=pl.lit(match.label, dtype=pl.UInt8),
        )
        for match in matches
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def _sample_context(
    states: pl.DataFrame, columns: tuple[str, ...], max_rows: int, seed: int
) -> pl.DataFrame:
    rows = (
        states.select(
            *columns,
            pl.when(pl.col("label") == 1)
            .then(pl.lit("archmother"))
            .otherwise(pl.lit("hidden_king"))
            .alias(WINNER),
        )
        .sample(fraction=1, shuffle=True, seed=seed % (1 << 64))
        .with_row_index("_row")
    )
    if rows.height <= max_rows:
        return rows.drop("_row")
    representatives = rows.unique(WINNER, keep="first", maintain_order=True)
    remainder = rows.join(
        representatives.select("_row"), on="_row", how="anti", maintain_order="left"
    )
    return pl.concat(
        [representatives, remainder.head(max_rows - representatives.height)]
    ).drop("_row")


def train_tabular(
    matches: Sequence[ReplayMatch],
    output: Path,
    *,
    model_version: str,
    device: str | None = None,
    seed: int = 42,
    bins: int = 10,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    context_rows: int = DEFAULT_CONTEXT_ROWS,
    num_estimators: int = DEFAULT_NUM_ESTIMATORS,
    features: FeatureSet = DEFAULT_FEATURE_SET,
    data_version: str | None = None,
) -> dict[str, ProbabilityMetrics]:
    """Fit a context from training matches, then evaluate held-out matches.

    Reuse the graph pipeline's whole-match split. The release bundle contains a
    bounded training context without identifiers and a manifest. Metrics and
    the summary use equal total weight per held-out match. The default feature
    set includes hero lineups and item counts and needs a matching data_version.
    """
    version = r"(0|[1-9][0-9]*)"
    if (
        re.fullmatch(rf"model/kumo-win/v{version}\.{version}\.{version}", model_version)
        is None
    ):
        raise ValueError("expected a model version such as model/kumo-win/v1.0.0")
    if features not in ("baseline", "heroes", "heroes-items"):
        raise ValueError(f"unknown Kumo feature set: {features}")
    if features == "heroes-items" and not data_version:
        raise ValueError("heroes-items requires a matching boon-data version")
    if context_rows < 2:
        raise ValueError("context_rows must be at least 2")
    if num_estimators < 1:
        raise ValueError("num_estimators must be at least 1")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    by_id = {match.match_id: match for match in matches}
    if len(by_id) != len(matches):
        raise ValueError("duplicate match IDs")
    if len(matches) < 4:
        raise ValueError(
            "Kumo needs at least four matches for training, validation, and test"
        )
    if any(
        match.label not in (0, 1)
        or not match.graphs
        or any(
            mid != match.match_id or not 0 <= tick < match.game_over_tick
            for mid, tick in match.snapshot_keys
        )
        for match in matches
    ):
        raise ValueError("matches need binary labels and pre-victory snapshots")
    splits = split_matches(
        pl.DataFrame({"match_id": list(by_id)}, schema={"match_id": pl.UInt64}),
        train_fraction,
        validation_fraction,
        seed=seed,
    )
    if len({by_id[mid].label for mid in splits.train["match_id"]}) != 2:
        raise ValueError(
            "training split needs both winning sides; use more matches or another seed"
        )
    train_matches = [by_id[mid] for mid in splits.train["match_id"]]
    items = (
        ItemCatalog.load(data_version)
        if features == "heroes-items" and data_version
        else None
    )
    lineups = (
        LineupFeatures.fit((graph for m in train_matches for graph in m.graphs), items)
        if features != "baseline"
        else LineupFeatures()
    )
    columns = feature_columns(lineups)
    training = _features(train_matches, lineups)
    context = _sample_context(training, columns, context_rows, seed)
    if items is not None and not any(
        context[name].null_count() < context.height
        for name in columns
        if "_item_" in name
    ):
        raise ValueError(
            "no complete inventories in the Kumo context; check the catalog version"
        )
    model = fit_kumo(
        context,
        seed=seed,
        device=device,
        num_estimators=num_estimators,
        columns=columns,
    )
    results = {}
    metrics = {}
    summary = [
        "# Deadlock win probability — KumoTabular small",
        "",
        (
            f"Model {model_version} uses {context.height} context rows from "
            f"{splits.train.height} training matches with {features} features."
        ),
        "",
        "| Split | Matches | Snapshots | Log loss | ECE | Brier score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, split in (("validation", splits.validation), ("test", splits.test)):
        states = _features([by_id[mid] for mid in split["match_id"]], lineups)
        scores = predict_probabilities(model, states, columns=columns)
        states = states.with_columns(probability=scores["archmother"])
        for period, rows in (
            (name, states),
            (
                f"{name}_early",
                states.filter(pl.col("elapsed_seconds") < EARLY_GAME_SECONDS),
            ),
        ):
            if rows.is_empty():
                continue
            # After time filtering each represented match still has equal weight.
            weights = rows.select(weight=1 / pl.len().over("match_id"))["weight"]
            result = probability_metrics(
                rows["label"].to_list(),
                rows["probability"].to_list(),
                bins=bins,
                sample_weight=weights.to_list(),
            )
            results[period] = result
            count = rows["match_id"].n_unique()
            metrics[period] = {
                "matches": count,
                "snapshots": rows.height,
                "log_loss": result.log_loss,
                "ece": result.ece,
                "brier_score": result.brier_score,
            }
            summary.append(
                f"| {period} | {count} | {rows.height} | {result.log_loss:.6f} | "
                f"{result.ece:.6f} | {result.brier_score:.6f} |"
            )

    summary.extend(
        ["", "Early-game scores use active replay time below 600 seconds.", ""]
    )
    output.mkdir(parents=True, exist_ok=False)
    context.write_parquet(output / CONTEXT_FILE)
    manifest = {
        "format_version": 1,
        "model_version": model_version,
        "architecture": "KumoTabular small",
        "target": WINNER,
        "labels": ["hidden_king", "archmother"],
        "structured_data_models": package_version("structured-data-models"),
        "features": columns,
        "feature_set": features,
        "hero_ids": lineups.hero_ids,
        "item_catalog": {"version": items.version, "sha256": items.checksum}
        if items
        else None,
        "context_rows": context.height,
        "seed": seed,
        "training_matches": splits.train.height,
        "num_estimators": num_estimators,
        "sha256": {CONTEXT_FILE: _sha256(output / CONTEXT_FILE)},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return results


def load_tabular_features(bundle: Path) -> LineupFeatures:
    """Restore the feature vocabulary and the verified item catalog for inference."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    catalog = manifest.get("item_catalog")
    items = ItemCatalog.load(catalog["version"]) if catalog is not None else None
    if items is not None and items.checksum != catalog["sha256"]:
        raise ValueError("KumoTabular item catalog checksum mismatch")
    lineups = LineupFeatures(tuple(manifest.get("hero_ids", ())), items)
    if manifest.get("features") != list(feature_columns(lineups)):
        raise ValueError("incompatible KumoTabular feature columns")
    return lineups


def load_tabular_model(bundle: Path, *, device: str | None = None) -> KumoTabular:
    """Verify and fit a downloaded Kumo release bundle."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format_version") != 1
        or manifest.get("architecture") != "KumoTabular small"
        or manifest.get("target") != WINNER
        or manifest.get("structured_data_models")
        != package_version("structured-data-models")
    ):
        raise ValueError("incompatible KumoTabular model manifest")
    with (bundle / CONTEXT_FILE).open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != manifest["sha256"][CONTEXT_FILE]:
            raise ValueError("KumoTabular context checksum mismatch")
        file.seek(0)
        context = pl.read_parquet(file)
    columns = tuple(manifest["features"])
    if context.columns != [*columns, WINNER]:
        raise ValueError("incompatible KumoTabular context columns")
    return fit_kumo(
        context,
        columns=columns,
        device=device,
        seed=manifest["seed"],
        num_estimators=manifest["num_estimators"],
    )
