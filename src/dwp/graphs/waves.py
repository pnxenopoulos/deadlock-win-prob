"""Summarize sampled lane troopers without adding graph nodes."""

import polars as pl

# Native playing-team and Midtown lane IDs.
_TEAM_LANES = tuple((team, lane) for team in (2, 3) for lane in (1, 4, 6))
_METRICS = ("count", "health", "front_distance")
WAVE_FEATURES = tuple(
    f"wave_{team}_{lane}_{metric}" for team, lane in _TEAM_LANES for metric in _METRICS
)


def wave_features(
    ticks: pl.DataFrame, troopers: pl.DataFrame, objectives: pl.DataFrame
) -> pl.DataFrame:
    """Return each side's wave count, summed HP, and closest range to the enemy Patron.

    Use same-tick troopers and the latest already-observed Patron as a stable
    enemy-base reference. The distance is horizontal, in world units. Empty
    waves have zero count and HP. Distance requires a wave and both positions.
    """
    bases = (
        objectives.filter(
            pl.col("objective_type") == "patron",
            pl.col("team_num").is_in([2, 3]),
            pl.col("x").is_finite(),
            pl.col("y").is_finite(),
        )
        .sort("observed_tick")
        .unique(subset=["tick", "team_num"], keep="last")
        .select(
            "tick",
            (5 - pl.col("team_num"))
            .cast(troopers.schema["team_num"])
            .alias("team_num"),
            pl.col("x").alias("enemy_x"),
            pl.col("y").alias("enemy_y"),
        )
    )
    observed = (
        troopers.filter(pl.col("trooper_type") == "trooper")
        .join(bases, on=["tick", "team_num"], how="left", validate="m:1")
        .with_columns(
            front_distance=(
                (pl.col("x") - pl.col("enemy_x")) ** 2
                + (pl.col("y") - pl.col("enemy_y")) ** 2
            ).sqrt()
        )
    )
    aggregates = []
    defaults = []
    for team, lane in _TEAM_LANES:
        selected = (pl.col("team_num") == team) & (pl.col("lane") == lane)
        count, health, distance = (f"wave_{team}_{lane}_{name}" for name in _METRICS)
        hp = pl.col("health").filter(selected)
        aggregates.extend(
            [
                selected.sum().alias(count),
                pl.when(hp.null_count() == 0).then(hp.sum()).alias(health),
                pl.col("front_distance").filter(selected).min().alias(distance),
            ]
        )
        defaults.extend(
            [
                pl.col(count).fill_null(0),
                pl.when(pl.col(count).fill_null(0) == 0)
                .then(0)
                .otherwise(pl.col(health))
                .alias(health),
            ]
        )
    return ticks.join(
        observed.group_by("tick").agg(aggregates), on="tick", how="left"
    ).with_columns(defaults)
