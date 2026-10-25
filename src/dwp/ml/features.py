"""Convert graph snapshots to tensors using training-only feature statistics."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Self

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from torch import Tensor

from dwp.graphs.snapshots import GraphSnapshot

logger = logging.getLogger(__name__)

NODE_TYPES = ("global", "player", "objective", "zipline")
_BASELINE = (
    "weapon_damage",
    "fire_rate",
    "move_speed",
    "stamina",
    "bullet_resist",
    "spirit_resist",
)
# team_num is encoded separately for every node, including players.
NUMERIC_FEATURES = (
    # Global
    (
        "elapsed_seconds",
        "mid_boss_up",
        "next_midboss",
    ),
    # Player
    (
        "x",
        "y",
        "z",
        "health",
        "max_health",
        "health_fraction",
        "barrier",
        "level",
        "souls",
        "spent_souls",
        "respawn_time",
        "is_alive",
        "health_regen",
        "has_rebirth",
        "has_rejuvenator",
        "has_ultimate_trained",
        "ultimate_cooldown_start",
        "ultimate_cooldown_end",
        "ap_net_worth",
        "gold_net_worth",
        "upgrade_count",
        "item_count",
        *(f"baseline_{stat}" for stat in _BASELINE),
    ),
    # Objective
    ("x", "y", "z", "health", "max_health", "health_fraction", "destroyed"),
    # Zipline
    ("x", "y", "z", "capturable", "cumulative_distance"),
)
CATEGORICAL_FEATURES = (
    # Global
    (),
    # Player
    ("hero_id",),
    # Objective
    ("objective_type", "lane", "phase"),
    # Zipline
    ("lane",),
)
# Native teams: missing/unknown, neutral, spectator, Hidden King, Archmother.
_TEAMS = {None: 0, 0: 1, 1: 2, 2: 3, 3: 4}


def _tables(graph: GraphSnapshot) -> tuple[pl.DataFrame, ...]:
    if graph.global_info.height != 1:
        raise ValueError("each graph must have exactly one global node")
    return graph.global_info, graph.players, graph.objectives, graph.ziplines


def _numeric(table: pl.DataFrame, columns: tuple[str, ...]) -> NDArray[np.float64]:
    """Explicit whitelist, derived counts/health fraction, and optional baselines."""
    derived: list[pl.Expr] = []
    if {"health", "max_health"} <= set(table.columns):
        derived.append(
            pl.when(pl.col("max_health") > 0)
            .then(pl.col("health") / pl.col("max_health"))
            .otherwise(None)
            .alias("health_fraction")
        )
    for source, target in (
        ("upgrade_ids", "upgrade_count"),
        ("item_ids", "item_count"),
    ):
        if source in table.columns:
            derived.append(pl.col(source).list.len().alias(target))
    if "baseline_stats" in table.columns:
        for stat in _BASELINE:
            derived.append(
                pl.col("baseline_stats")
                .list.eval(
                    pl.element()
                    .filter(pl.element().struct.field("stat") == stat)
                    .struct.field("value")
                )
                .list.first()
                .alias(f"baseline_{stat}")
            )
    table = table.with_columns(derived)
    values = (
        table.select(
            pl.col(name).cast(pl.Float64, strict=False)
            if name in table.columns
            else pl.lit(None, dtype=pl.Float64).alias(name)
            for name in columns
        )
        .to_numpy()
        .astype(np.float64)
    )
    # Polars literal-only selections can have one row even for empty tables.
    values = values[: table.height]
    values[~np.isfinite(values)] = np.nan
    return np.sign(values) * np.log1p(np.abs(values))


@dataclass(frozen=True, slots=True)
class GraphTensors:
    """One graph in global/player/objective/zipline order; no labels or IDs.

    Numerical tensors contain standardized values followed by observed masks.
    Categorical indices use 0=missing/padding, 1=unseen, 2+=training vocabulary.
    teams contains categorical team IDs for every node, including players.
    Positions use game coordinates / 10,000 for pairwise attention geometry.
    """

    numeric: tuple[Tensor, ...]
    categorical: tuple[Tensor, ...]
    upgrades: Tensor
    node_types: Tensor
    teams: Tensor
    positions: Tensor
    position_known: Tensor
    connections: Tensor
    connections_known: Tensor

    def to(self, device: torch.device | str) -> Self:
        """Copy all feature tensors to the model's device."""
        return replace(
            self,
            numeric=tuple(t.to(device) for t in self.numeric),
            categorical=tuple(t.to(device) for t in self.categorical),
            upgrades=self.upgrades.to(device),
            node_types=self.node_types.to(device),
            teams=self.teams.to(device),
            positions=self.positions.to(device),
            position_known=self.position_known.to(device),
            connections=self.connections.to(device),
            connections_known=self.connections_known.to(device),
        )


@dataclass(frozen=True, slots=True)
class GraphEncoder:
    """Training-fitted normalization/vocabularies; reuse unchanged on held-out data.

    Keep this object with the model checkpoint. Only whitelisted state fields
    are used: account/match/entity IDs, raw ticks, and outcomes are excluded.
    Upgrade tokens pool both items and abilities, including without a catalog.
    """

    means: tuple[NDArray[np.float64], ...]
    scales: tuple[NDArray[np.float64], ...]
    vocabularies: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, graphs: Iterable[GraphSnapshot]) -> Self:
        """Fit in one pass over training graphs only, ignoring missing values."""
        sums = [np.zeros(len(names)) for names in NUMERIC_FEATURES]
        squares = [np.zeros(len(names)) for names in NUMERIC_FEATURES]
        counts = [np.zeros(len(names)) for names in NUMERIC_FEATURES]
        categories: dict[str, set[str]] = {
            name: set() for names in CATEGORICAL_FEATURES for name in names
        }
        categories["upgrade_ids"] = set()
        seen = False
        for index, graph in enumerate(graphs, 1):
            seen = True
            for i, table in enumerate(_tables(graph)):
                values = _numeric(table, NUMERIC_FEATURES[i])
                sums[i] += np.nansum(values, axis=0)
                squares[i] += np.nansum(values**2, axis=0)
                counts[i] += np.isfinite(values).sum(axis=0)
                for name in CATEGORICAL_FEATURES[i]:
                    if name in table.columns:
                        categories[name].update(
                            str(v) for v in table[name].drop_nulls()
                        )
            if "upgrade_ids" in graph.players.columns:
                categories["upgrade_ids"].update(
                    str(v) for v in graph.players["upgrade_ids"].explode().drop_nulls()
                )
            if index % 10_000 == 0:
                logger.info("Fitted feature statistics on %d training snapshots", index)
        if not seen:
            raise ValueError("fit requires at least one training graph")
        means = tuple(
            total / np.maximum(count, 1) for total, count in zip(sums, counts)
        )
        scales = tuple(
            np.where(
                count > 1,
                np.sqrt(np.maximum(square / np.maximum(count, 1) - mean**2, 0)),
                1,
            )
            for square, count, mean in zip(squares, counts, means)
        )
        return cls(
            means,
            tuple(np.where(scale < 1e-6, 1, scale) for scale in scales),
            {
                name: {value: i + 2 for i, value in enumerate(sorted(values))}
                for name, values in categories.items()
            },
        )

    def transform(self, graph: GraphSnapshot) -> GraphTensors:
        """Encode without fitting, downloading catalogs, or materializing edges."""
        tables = _tables(graph)
        numeric: list[Tensor] = []
        categorical: list[Tensor] = []
        for i, table in enumerate(tables):
            values = _numeric(table, NUMERIC_FEATURES[i])
            known = np.isfinite(values)
            scaled = np.where(known, (values - self.means[i]) / self.scales[i], 0)
            numeric.append(
                torch.tensor(
                    np.concatenate([scaled, known], axis=1), dtype=torch.float32
                )
            )
            categorical.append(
                torch.tensor(
                    [
                        [
                            0
                            if row.get(name) is None
                            else self.vocabularies[name].get(str(row[name]), 1)
                            for name in CATEGORICAL_FEATURES[i]
                        ]
                        for row in table.iter_rows(named=True)
                    ],
                    dtype=torch.long,
                ).reshape(table.height, len(CATEGORICAL_FEATURES[i]))
            )
        tokens: list[list[int] | None] = (
            graph.players["upgrade_ids"].to_list()
            if "upgrade_ids" in graph.players.columns
            else [None] * graph.players.height
        )
        upgrades = torch.zeros(
            (len(tokens), max(1, max((len(t or []) for t in tokens), default=0))),
            dtype=torch.long,
        )
        for row, owned in enumerate(tokens):
            if owned:
                upgrades[row, : len(owned)] = torch.tensor(
                    [self.vocabularies["upgrade_ids"].get(str(t), 1) for t in owned]
                )
        nodes = graph.nodes
        raw_positions = torch.tensor(
            nodes.select("x", "y", "z").to_numpy().copy(), dtype=torch.float32
        )
        position_known = torch.isfinite(raw_positions).all(dim=1)
        positions = torch.where(position_known[:, None], raw_positions / 10000, 0)
        connections = torch.zeros((nodes.height, nodes.height), dtype=torch.bool)
        connections_known = torch.zeros(nodes.height, dtype=torch.bool)
        offset = sum(table.height for table in tables[:3])
        zip_rows = list(graph.ziplines.iter_rows(named=True))
        handles = {row.get("node_handle"): offset + i for i, row in enumerate(zip_rows)}
        for i, row in enumerate(zip_rows):
            neighbors = row.get("connections")
            connections_known[offset + i] = neighbors is not None
            for target in neighbors or []:
                if target in handles:
                    connections[offset + i, handles[target]] = True
        return GraphTensors(
            numeric=tuple(numeric),
            categorical=tuple(categorical),
            upgrades=upgrades,
            node_types=torch.cat(
                [
                    torch.full((t.height,), i, dtype=torch.long)
                    for i, t in enumerate(tables)
                ]
            ),
            teams=torch.tensor(
                [_TEAMS.get(t, 0) for t in nodes["team_num"]], dtype=torch.long
            ),
            positions=positions,
            position_known=position_known,
            connections=connections,
            connections_known=connections_known,
        )
