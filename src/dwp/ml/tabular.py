"""Aggregate replay snapshots and score them with KumoTabular small."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import polars as pl

from dwp.graphs.snapshots import GraphSnapshot
from dwp.ml.tabular_features import LineupFeatures

if TYPE_CHECKING:
    import torch
    from sdm import TableTensor
    from sdm.models import KumoTabular

_BASELINE = LineupFeatures()
WINNER = "winner"
SIDES = {2: "hidden_king", 3: "archmother"}
_OBJECTIVES = ("walker", "barracks", "shrine", "patron")
_PLAYER_FEATURES = (
    "souls",
    "spent_souls",
    "gold_net_worth",
    "alive",
    "health_fraction",
    "mean_level",
)
_TEAM_FEATURES = (
    *_PLAYER_FEATURES,
    *(
        f"{kind}_{field}"
        for kind in _OBJECTIVES
        for field in ("standing", "health_fraction")
    ),
    "patron_phase",
)
FEATURE_COLUMNS = (
    "elapsed_seconds",
    "mid_boss_up",
    *(f"{side}_{field}" for side in SIDES.values() for field in _TEAM_FEATURES),
    "gold_net_worth_diff",
)


def _health_fraction() -> pl.Expr:
    return (
        pl.when(pl.col("max_health") > 0)
        .then((pl.col("health") / pl.col("max_health")).clip(0, 1))
        .otherwise(None)
    )


def _sum_observed(column: str) -> pl.Expr:
    value = pl.col(column)
    return pl.when(value.count() > 0).then(value.sum()).otherwise(None)


def feature_columns(lineups: LineupFeatures) -> tuple[str, ...]:
    """Return model columns in their fixed training and inference order."""
    return (
        *FEATURE_COLUMNS,
        *(f"{side}_{field}" for side in SIDES.values() for field in lineups.fields),
    )


def tabular_features(
    graphs: Sequence[GraphSnapshot], lineups: LineupFeatures = _BASELINE
) -> pl.DataFrame:
    """Return one row per snapshot with match ID, tick, and fixed model features.

    Economy totals, mean hero health fraction, alive counts, and objective
    summaries are computed separately for each team. Dead heroes count as zero
    health. Missing observations stay null. The net worth difference is
    Archmother minus Hidden King. The elapsed clock excludes pauses. Match ID
    and tick are available for joins and are excluded from Kumo inputs.
    Pass the training lineup vocabulary and item catalog to add those features.
    """
    rows = []
    for graph in graphs:
        row = graph.global_info.select("elapsed_seconds", "mid_boss_up").row(
            0, named=True
        )
        row.update(match_id=graph.match_id, tick=graph.tick)
        players = graph.players
        objectives = graph.objectives
        for team, side in SIDES.items():
            team_players = players.filter(pl.col("team_num") == team)
            stats = team_players.select(
                *(
                    _sum_observed(column).alias(column)
                    for column in ("souls", "spent_souls", "gold_net_worth")
                ),
                alive=_sum_observed("is_alive"),
                health_fraction=pl.when(~pl.col("is_alive"))
                .then(0.0)
                .otherwise(_health_fraction())
                .mean(),
                mean_level=pl.col("level").mean(),
            ).row(0, named=True)
            objective_type = pl.col("objective_type")
            objective_stats = []
            for kind in _OBJECTIVES:
                selected = objective_type == kind
                health = pl.col("health").filter(selected)
                objective_stats.extend(
                    (
                        pl.when(health.count() > 0)
                        .then((health > 0).sum())
                        .alias(f"{kind}_standing"),
                        _health_fraction()
                        .filter(selected)
                        .mean()
                        .alias(f"{kind}_health_fraction"),
                    )
                )
            stats.update(
                objectives.filter(pl.col("team_num") == team)
                .select(
                    *objective_stats,
                    patron_phase=pl.col("phase")
                    .filter(objective_type == "patron")
                    .max(),
                )
                .row(0, named=True)
            )
            stats.update(lineups.values(team_players))
            row.update({f"{side}_{name}": value for name, value in stats.items()})
        rows.append(row)
    columns = feature_columns(lineups)
    schema = {
        "match_id": pl.UInt64,
        "tick": pl.Int64,
        **{name: pl.Float64 for name in columns if name != "gold_net_worth_diff"},
    }
    return (
        pl.DataFrame(rows, schema=schema)
        .with_columns(
            gold_net_worth_diff=pl.col("archmother_gold_net_worth")
            - pl.col("hidden_king_gold_net_worth")
        )
        .select("match_id", "tick", *columns)
    )


def _table(frame: pl.DataFrame, device: torch.device) -> TableTensor:
    from sdm import TableTensor

    stypes: dict[str, Literal["categorical", "numerical"]] = {
        name: "categorical" if name == WINNER else "numerical" for name in frame.columns
    }
    return TableTensor.from_arrow(frame.to_arrow(), stypes=stypes, device=device)


def fit_kumo(
    context: pl.DataFrame,
    *,
    device: str | None = None,
    seed: int = 42,
    num_estimators: int = 8,
    columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> KumoTabular:
    """Fit KumoTabular small with a context containing both winning sides."""
    import torch
    from sdm.models import KumoTabular

    if set(context[WINNER].unique()) != set(SIDES.values()):
        raise ValueError(
            "Kumo context needs both winning sides; use more training matches"
        )
    if num_estimators < 1:
        raise ValueError("num_estimators must be positive")
    model_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    table = _table(context.select(*columns, WINNER), model_device)
    model = KumoTabular(task="classification", size="small", device=model_device)
    model.fit(
        x=table.drop_columns(WINNER),
        y=table[WINNER],
        num_estimators=num_estimators,
        generator=torch.Generator(device=model_device).manual_seed(seed % (1 << 64)),
    )
    return model


def predict_probabilities(
    model: KumoTabular,
    states: pl.DataFrame,
    *,
    columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pl.DataFrame:
    """Return two probability columns, one for each team, in input row order."""
    probabilities = []
    for batch in states.select(columns).iter_slices(n_rows=256):
        if not batch.is_empty():
            query = _table(batch, next(model.parameters()).device)
            scores = cast(pl.DataFrame, pl.from_arrow(model.predict(query).to_arrow()))
            probabilities.append(scores.select(*SIDES.values()))
    return (
        pl.concat(probabilities)
        if probabilities
        else pl.DataFrame(schema=dict.fromkeys(SIDES.values(), pl.Float64))
    )
