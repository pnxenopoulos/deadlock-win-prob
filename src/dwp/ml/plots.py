"""Predict and plot a demo's win probabilities at regular game-time intervals."""

from bisect import bisect_left
from math import floor, isfinite

import polars as pl
import torch
from boon import Demo
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from dwp.graphs.snapshots import iter_graphs
from dwp.ml.features import GraphEncoder
from dwp.ml.model import HeterogeneousGraphTransformer


@torch.inference_mode()
def predict_win_probabilities(
    demo: Demo,
    model: HeterogeneousGraphTransformer,
    encoder: GraphEncoder,
    *,
    interval_seconds: float = 10,
    data_version: str | None = None,
) -> pl.DataFrame:
    """Return tick, clocks, and both teams' probabilities every 10 active seconds.

    Reuse the encoder and optional data_version used for training. Sampling
    starts at the first known active recorded tick. Binary search over active
    ticks finds the first tick at/after each pause-adjusted recording-time
    target, including when a target coincides with a pause. Predictions stop
    before game over, or at recording end for unfinished/partial demos.
    States without player observations are skipped. Match-clock labels are
    reconstructed only after inference; they never select ticks or enter the
    model. Inference uses the model's device and restores its mode.
    """
    if not isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be finite and positive")
    end_tick = demo.total_ticks
    if demo.game_over_tick is not None:
        end_tick = min(end_tick, demo.game_over_tick - 1)
    if end_tick < 0:
        raise ValueError("demo has no pre-game-over ticks")
    active_ticks: list[int] = (
        demo.world_ticks.filter(
            ~pl.col("is_paused"), pl.col("tick").is_between(0, end_tick)
        )["tick"]
        .sort()
        .to_list()
    )
    if not active_ticks:
        raise ValueError("demo has no known active recorded ticks")
    start_seconds = demo.tick_to_seconds(active_ticks[0])
    duration = demo.tick_to_seconds(active_ticks[-1]) - start_seconds
    if not isfinite(duration) or duration < 0:
        raise ValueError("demo has no usable time interval")
    ticks = list(
        dict.fromkeys(
            active_ticks[
                bisect_left(
                    active_ticks,
                    start_seconds + i * interval_seconds,
                    key=demo.tick_to_seconds,
                )
            ]
            for i in range(floor(duration / interval_seconds) + 1)
        )
    )
    device = next(model.parameters()).device
    was_training = model.training
    rows = []
    model.eval()
    try:
        for graph in iter_graphs(
            demo, ticks=ticks, end_tick=end_tick, data_version=data_version
        ):
            if graph.players.is_empty():
                continue
            probability = model(encoder.transform(graph).to(device)).sigmoid().item()
            if not isfinite(probability):
                raise ValueError("model produced a non-finite win probability")
            rows.append(
                {
                    "tick": graph.tick,
                    "elapsed_seconds": graph.global_info["elapsed_seconds"][0],
                    "archmother_win_probability": probability,
                    "hidden_king_win_probability": 1 - probability,
                }
            )
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("demo has no sampled states with player observations")
    # Boon's match-clock offset uses game-over data: display metadata only.
    return pl.DataFrame(
        [
            {**row, "match_time_seconds": demo.tick_to_match_seconds(row["tick"])}
            for row in rows
        ],
        schema={
            "tick": pl.Int64,
            "elapsed_seconds": pl.Float64,
            "match_time_seconds": pl.Float64,
            "archmother_win_probability": pl.Float64,
            "hidden_king_win_probability": pl.Float64,
        },
    )


def plot_win_probability(
    demo: Demo,
    model: HeterogeneousGraphTransformer,
    encoder: GraphEncoder,
    *,
    interval_seconds: float = 10,
    data_version: str | None = None,
) -> Figure:
    """Return a saveable figure of both teams' predicted win probabilities.

    Uses match time in minutes, falling back to pause-adjusted recording time
    when match time is unavailable. No winner endpoint is added to the curve.
    """
    predictions = predict_win_probabilities(
        demo,
        model,
        encoder,
        interval_seconds=interval_seconds,
        data_version=data_version,
    )
    match_clock = predictions["match_time_seconds"].null_count() == 0
    time = predictions["match_time_seconds" if match_clock else "elapsed_seconds"] / 60
    figure = Figure(figsize=(10, 4), layout="constrained")
    axes = figure.subplots()
    axes.plot(
        time,
        predictions["archmother_win_probability"],
        label="Archmother",
        color="tab:blue",
        marker=".",
        markersize=3,
    )
    axes.plot(
        time,
        predictions["hidden_king_win_probability"],
        label="Hidden King",
        color="tab:orange",
        marker=".",
        markersize=3,
    )
    axes.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    axes.set(
        xlabel="Match time (minutes)"
        if match_clock
        else "Recorded time (minutes, pauses excluded)",
        ylabel="Win probability",
        ylim=(-0.02, 1.02),
        yticks=[0, 0.25, 0.5, 0.75, 1],
        title=f"Match {demo.match_id}: win probability"
        if demo.match_id is not None
        else "Win probability over time",
    )
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes.grid(axis="y", alpha=0.2)
    axes.legend()
    return figure
