"""Pause-aware demo sampling, inference-only execution, and plot export."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
import torch
from boon import Demo

from dwp.features import GraphEncoder
from dwp.model import HeterogeneousGraphTransformer
from dwp.plots import plot_win_probability, predict_win_probabilities
from dwp.representations import GraphSnapshot


@pytest.fixture(autouse=True, scope="module")
def small_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def timeline(
    graph_snapshot: GraphSnapshot, monkeypatch: pytest.MonkeyPatch
) -> tuple[Mock, HeterogeneousGraphTransformer, GraphEncoder, Mock]:
    demo = Mock(spec=Demo)
    demo.match_id = graph_snapshot.match_id
    demo.total_ticks = 350
    demo.game_over_tick = 301
    demo.pregame_seconds = 2.0
    # A five-second pause at ticks 50..100, at 10 ticks/sec.
    demo.tick_to_seconds.side_effect = lambda t: (t - min(max(t - 50, 0), 50)) / 10

    def graphs(demo, *, ticks, **kwargs):
        for tick in ticks:
            elapsed = demo.tick_to_seconds(tick)
            match_time = (
                None if demo.pregame_seconds is None else elapsed - demo.pregame_seconds
            )
            yield replace(
                graph_snapshot,
                tick=tick,
                global_info=graph_snapshot.global_info.with_columns(
                    pl.lit(elapsed).alias("elapsed_seconds"),
                    pl.lit(match_time, dtype=pl.Float64).alias("match_time_seconds"),
                ),
            )

    snapshots = Mock(side_effect=graphs)
    monkeypatch.setattr("dwp.plots.iter_graphs", snapshots)
    encoder = GraphEncoder.fit([graph_snapshot])
    model = HeterogeneousGraphTransformer(
        encoder, hidden_dim=16, num_layers=1, dropout=0.5
    )
    return demo, model, encoder, snapshots


def test_ten_second_samples_use_active_time_and_preserve_model_mode(
    timeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo, model, encoder, snapshots = timeline
    monkeypatch.setattr(
        GraphEncoder,
        "fit",
        Mock(side_effect=AssertionError("must reuse training encoder")),
    )
    modes = []
    handle = model.register_forward_pre_hook(
        lambda m, inputs: modes.append((m.training, torch.is_grad_enabled()))
    )
    result = predict_win_probabilities(
        demo, model, encoder, data_version="training-version"
    )
    handle.remove()
    assert snapshots.call_args.kwargs == {
        "ticks": [20, 170, 270],
        "end_tick": 300,
        "data_version": "training-version",
    }
    assert result["tick"].to_list() == [20, 170, 270]
    assert result["elapsed_seconds"].to_list() == [2, 12, 22]
    assert result["match_time_seconds"].to_list() == [0, 10, 20]
    assert modes == [(False, False)] * 3
    assert model.training and all(p.grad is None for p in model.parameters())
    np.testing.assert_allclose(
        result["archmother_win_probability"].to_numpy()
        + result["hidden_king_win_probability"].to_numpy(),
        1,
    )
    assert result["archmother_win_probability"].is_between(0, 1).all()


def test_game_over_cutoff_does_not_add_a_winner_endpoint(timeline) -> None:
    demo, model, encoder, snapshots = timeline
    demo.game_over_tick = 270
    result = predict_win_probabilities(demo, model, encoder)
    assert result["tick"].to_list() == [20, 170]
    assert snapshots.call_args.kwargs["end_tick"] == 269


def test_partial_recording_uses_recording_clock(timeline) -> None:
    demo, model, encoder, snapshots = timeline
    demo.pregame_seconds = None
    demo.game_over_tick = None
    result = predict_win_probabilities(demo, model.eval(), encoder)
    assert result["tick"].to_list() == [0, 150, 250, 350]
    assert result["elapsed_seconds"].to_list() == [0, 10, 20, 30]
    assert result["match_time_seconds"].null_count() == 4
    assert not model.training
    assert snapshots.call_args.kwargs["end_tick"] == 350


def test_pause_during_pregame_is_handled(timeline) -> None:
    demo, model, encoder, _snapshots = timeline
    demo.pregame_seconds = 7
    result = predict_win_probabilities(demo, model, encoder)
    assert result["tick"].to_list() == [120, 220]
    assert result["match_time_seconds"].to_list() == [0, 10]


def test_plot_exports_both_teams_and_uses_the_available_clock(
    timeline, tmp_path: Path
) -> None:
    demo, model, encoder, _snapshots = timeline
    figure = plot_win_probability(demo, model, encoder)
    axes = figure.axes[0]
    assert axes.get_xlabel() == "Match time (minutes)"
    assert [line.get_label() for line in axes.lines[:2]] == [
        "Archmother",
        "Hidden King",
    ]
    np.testing.assert_allclose(np.asarray(axes.lines[0].get_xdata()), [0, 1 / 6, 1 / 3])
    figure.savefig(tmp_path / "win-probability.png")
    assert (tmp_path / "win-probability.png").stat().st_size > 1000
    demo.pregame_seconds = None
    assert (
        "Recorded time"
        in plot_win_probability(demo, model, encoder).axes[0].get_xlabel()
    )


def test_skips_empty_player_states(timeline) -> None:
    demo, model, encoder, snapshots = timeline
    original = snapshots.side_effect

    def missing_first(*args, **kwargs):
        for i, graph in enumerate(original(*args, **kwargs)):
            yield replace(graph, players=graph.players.clear()) if i == 0 else graph

    snapshots.side_effect = missing_first
    assert predict_win_probabilities(demo, model, encoder)["tick"].to_list() == [
        170,
        270,
    ]


def test_empty_samples_and_inference_errors_restore_mode(
    timeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo, model, encoder, snapshots = timeline
    snapshots.return_value = iter([])
    snapshots.side_effect = None
    with pytest.raises(ValueError, match="no sampled states"):
        predict_win_probabilities(demo, model, encoder)
    assert model.training
    snapshots.side_effect = RuntimeError("could not decode")
    with pytest.raises(RuntimeError, match="could not decode"):
        predict_win_probabilities(demo, model, encoder)
    assert model.training


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_intervals_are_rejected(timeline, interval: float) -> None:
    demo, model, encoder, snapshots = timeline
    with pytest.raises(ValueError, match="interval_seconds"):
        predict_win_probabilities(demo, model, encoder, interval_seconds=interval)
    snapshots.assert_not_called()


def test_interval_is_configurable(timeline) -> None:
    demo, model, encoder, _snapshots = timeline
    assert predict_win_probabilities(demo, model, encoder, interval_seconds=5)[
        "tick"
    ].to_list() == [20, 120, 170, 220, 270]


def test_single_sample_remains_visible(timeline) -> None:
    demo, model, encoder, _snapshots = timeline
    demo.game_over_tick = 40
    figure = plot_win_probability(demo, model, encoder)
    assert np.asarray(figure.axes[0].lines[0].get_xdata()).size == 1
    assert figure.axes[0].lines[0].get_marker() == "."


def test_nonfinite_predictions_fail_without_changing_mode(timeline) -> None:
    demo, model, encoder, _snapshots = timeline
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        predict_win_probabilities(demo, model, encoder)
    assert model.training
