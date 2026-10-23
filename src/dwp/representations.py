"""Sample Boon demos into heterogeneous graph snapshots without future state."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import polars as pl
from boon import Demo, HeroStat, StatMode, data

_NODE_COLUMNS = ("node_id", "node_type", "team_num", "x", "y", "z")
_STAT_SCHEMA = {
    "stat": pl.String,
    "value": pl.Float64,
    "unit": pl.String,
    "mode": pl.String,
    "ruleset": pl.String,
    "status": pl.String,
    "diagnostic": pl.String,
}
_BASELINE_DTYPE = pl.List(pl.Struct(_STAT_SCHEMA))
_DEFAULT_STATS = (
    HeroStat.WEAPON_DAMAGE,
    HeroStat.FIRE_RATE,
    HeroStat.MOVE_SPEED,
    HeroStat.STAMINA,
    HeroStat.BULLET_RESIST,
    HeroStat.SPIRIT_RESIST,
)


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """One sampled state, with separate feature tables for each node type.

    Node IDs identify accounts, objective lifetimes, and full zipline handles.
    Unknown/ambiguous player identities receive frame-local IDs. Edges are
    materialized on demand because complete graphs grow quadratically.
    """

    tick: int
    match_id: int | None
    build: int
    data_version: str | None
    global_info: pl.DataFrame
    players: pl.DataFrame
    objectives: pl.DataFrame
    ziplines: pl.DataFrame

    @property
    def nodes(self) -> pl.DataFrame:
        """Common node index; type-specific features remain in the four tables."""
        return pl.concat(
            table.select(_NODE_COLUMNS)
            for table in (
                self.global_info,
                self.players,
                self.objectives,
                self.ziplines,
            )
        )

    @property
    def edges(self) -> pl.DataFrame:
        """All directed pairs except self-loops, with type/team labels and distance."""
        nodes = self.nodes
        same_team = pl.col("team_num") == pl.col("team_num_target")
        playing_teams = pl.col("team_num").is_in([2, 3]) & pl.col(
            "team_num_target"
        ).is_in([2, 3])
        return (
            nodes.join(nodes, how="cross", suffix="_target")
            .filter(pl.col("node_id") != pl.col("node_id_target"))
            .select(
                source=pl.col("node_id"),
                target=pl.col("node_id_target"),
                source_type=pl.col("node_type"),
                target_type=pl.col("node_type_target"),
                source_team=pl.col("team_num"),
                target_team=pl.col("team_num_target"),
                team_relation=pl.when(
                    (pl.col("node_type") == "global")
                    | (pl.col("node_type_target") == "global")
                )
                .then(pl.lit("global"))
                .when(playing_teams & same_team)
                .then(pl.lit("same_team"))
                .when(playing_teams & ~same_team)
                .then(pl.lit("opposing_team"))
                .when((pl.col("team_num") == 0) | (pl.col("team_num_target") == 0))
                .then(pl.lit("neutral"))
                .otherwise(pl.lit("unknown")),
                distance=pl.sum_horizontal(
                    [
                        (pl.col(axis) - pl.col(f"{axis}_target")) ** 2
                        for axis in ("x", "y", "z")
                    ],
                    ignore_nulls=False,
                ).sqrt(),
            )
        )


def _nodes(frame: pl.DataFrame, kind: str, node_id: pl.Expr) -> pl.DataFrame:
    """Give each feature table a common, nullable node-index schema."""
    return frame.with_columns(
        node_id=node_id,
        node_type=pl.lit(kind),
        **{
            name: pl.col(name).cast(dtype)
            if name in frame.columns
            else pl.lit(None, dtype=dtype)
            for name, dtype in {
                "team_num": pl.Int64,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            }.items()
        },
    )


def _at_ticks(
    events: pl.DataFrame, ticks: pl.DataFrame, keys: Sequence[str] = ()
) -> pl.DataFrame:
    """Carry observations backward to query ticks, never using future rows."""
    history = events.with_columns(pl.col("tick").alias("observed_tick")).sort(
        "tick", maintain_order=True
    )
    queries = ticks
    if keys:
        queries = ticks.join(events.select(keys).unique(), how="cross").sort("tick")
    sampled = queries.join_asof(
        history,
        on="tick",
        by=list(keys) or None,
        strategy="backward",
        check_sortedness=False,
    )
    return sampled.filter(pl.col("observed_tick").is_not_null()) if keys else sampled


def iter_graphs(
    demo: Demo,
    every: int = 64,
    *,
    ticks: Sequence[int] | None = None,
    start_tick: int | None = None,
    end_tick: int | None = None,
    data_version: str | None = None,
    baseline_stats: Sequence[HeroStat | str] = _DEFAULT_STATS,
) -> Iterator[GraphSnapshot]:
    """Yield one graph at every N recorded ticks, within an optional window.

    Explicit ticks override every; start/end windows still apply.
    Recorded player fields, objective health/phase, and zipline geometry/control
    are retained. Timers keep Boon's raw source values. Match time is null when
    the recording does not establish the game start; elapsed_seconds is the
    pause-adjusted recording time.

    upgrade_ids always retains the recorded tokens (items and abilities).
    An explicit data_version classifies item_ids and adds baseline_stats with
    value/unit/status/diagnostic. Baseline includes passive item and permanent
    bonuses, not just untouched hero stats. Without a version those fields are
    null. Catalogs may be downloaded by Boon. No match outcome is included.
    """
    if every <= 0:
        raise ValueError("every must be positive")
    if data_version is not None and not data_version:
        raise ValueError("data_version must be a nonempty version string")
    frames = cast(
        dict[str, pl.DataFrame],
        demo.snapshots(
            ["player_ticks", "world_ticks", "ziplines"],
            every=every if ticks is None else None,
            ticks=list(ticks) if ticks is not None else None,
            start_tick=start_tick,
            end_tick=end_tick,
        ),
    )
    tick_frame = (
        pl.concat(frame.select("tick") for frame in frames.values())
        .unique()
        .sort("tick")
    )
    if tick_frame.is_empty():
        return
    tick_values: list[int] = tick_frame["tick"].to_list()
    roster = demo.players.select("steam_id", "team_num").filter(
        pl.col("steam_id").is_not_null() & pl.col("steam_id").is_unique()
    )
    players = (
        frames["player_ticks"]
        .with_row_index("_row")
        .join(roster, on="steam_id", how="left", validate="m:1", maintain_order="left")
        .with_columns(
            identity_known=pl.col("steam_id").is_not_null()
            & pl.struct("tick", "steam_id").is_unique(),
            item_ids=pl.lit(None, dtype=pl.List(pl.UInt32)),
            baseline_stats=pl.lit(None, dtype=_BASELINE_DTYPE),
        )
    )
    if data_version is not None:
        catalog = json.loads(
            data.catalog_path("abilities", data_version).read_text(encoding="utf-8")
        )
        item_ids: list[int] = [
            record["ability_id"]
            for record in catalog["records"]
            if record.get("definition", {}).get("m_eAbilityType") == "EAbilityType_Item"
        ]
        players = players.with_columns(
            item_ids=pl.col("upgrade_ids").list.eval(
                pl.element().filter(pl.element().is_in(item_ids))
            )
        )
        steam_ids: list[int] = (
            players.filter(pl.col("identity_known"))["steam_id"].unique().to_list()
        )
        if baseline_stats and steam_ids:
            calculated = demo.calculate_hero_stats(
                ticks=tick_values,
                steam_ids=steam_ids,
                data_version=data_version,
                stats=baseline_stats,
                mode=StatMode.BASELINE,
                strict=False,
            ).values
            calculated = calculated.group_by("tick", "steam_id", "hero_id").agg(
                pl.struct(list(_STAT_SCHEMA)).alias("baseline_stats")
            )
            players = (
                players.drop("baseline_stats")
                .join(
                    calculated,
                    on=["tick", "steam_id", "hero_id"],
                    how="left",
                    validate="m:1",
                    maintain_order="left",
                )
                .with_columns(
                    pl.when(pl.col("identity_known"))
                    .then(pl.col("baseline_stats"))
                    .otherwise(None)
                    .cast(_BASELINE_DTYPE)
                    .alias("baseline_stats")
                )
            )
    players = _nodes(
        players,
        "player",
        pl.when(pl.col("identity_known"))
        .then(pl.concat_str(pl.lit("player:"), pl.col("steam_id")))
        .otherwise(
            pl.concat_str(
                pl.lit("player:unknown:"), pl.col("tick"), pl.lit(":"), pl.col("_row")
            )
        ),
    ).drop("_row")
    objectives = _nodes(
        _at_ticks(demo.objectives, tick_frame, ("entity_id", "entity_serial")),
        "objective",
        pl.concat_str(
            pl.lit("objective:"),
            pl.col("entity_id"),
            pl.lit(":"),
            pl.col("entity_serial"),
        ),
    ).with_columns(destroyed=pl.col("health") <= 0)
    ziplines = _nodes(
        frames["ziplines"],
        "zipline",
        pl.concat_str(pl.lit("zipline:"), pl.col("node_handle")),
    )
    boss = _at_ticks(
        demo.mid_boss.filter(pl.col("event").is_in(["spawned", "killed"])), tick_frame
    )
    global_info = _nodes(
        tick_frame.join(
            frames["world_ticks"],
            on="tick",
            how="left",
            maintain_order="left",
            validate="1:1",
        )
        .join(
            boss.select("tick", "event"),
            on="tick",
            how="left",
            maintain_order="left",
            validate="1:1",
        )
        .with_columns(
            pl.Series(
                "elapsed_seconds",
                [demo.tick_to_seconds(tick) for tick in tick_values],
                dtype=pl.Float64,
            ),
            pl.Series(
                "match_time_seconds",
                [demo.tick_to_match_seconds(tick) for tick in tick_values],
                dtype=pl.Float64,
            ),
            pl.when(pl.col("event") == "spawned")
            .then(True)
            .when(pl.col("event") == "killed")
            .then(False)
            .otherwise(None)
            .alias("mid_boss_up"),
        )
        .drop("event"),
        "global",
        pl.lit("global"),
    )
    tables = (global_info, players, objectives, ziplines)
    groups = [table.partition_by("tick", as_dict=True) for table in tables]
    for tick in tick_values:
        yield GraphSnapshot(
            tick=tick,
            match_id=demo.match_id,
            build=demo.build,
            data_version=data_version,
            global_info=groups[0][(tick,)],
            players=groups[1].get((tick,), players.clear()),
            objectives=groups[2].get((tick,), objectives.clear()),
            ziplines=groups[3].get((tick,), ziplines.clear()),
        )
