"""Known probability scores, bin boundaries, weighted ECE, and plot export."""

from math import log
from pathlib import Path

import numpy as np
import pytest

from dwp.ml.metrics import plot_calibration, probability_metrics


def test_known_scores() -> None:
    result = probability_metrics([0, 1], [0.25, 0.75], bins=2)
    assert result.log_loss == pytest.approx(-log(0.75))
    assert result.ece == pytest.approx(0.25)
    assert result.brier_score == pytest.approx(0.0625)
    assert result.calibration["count"].to_list() == [1, 1]
    assert result.calibration["mean_probability"].to_list() == [0.25, 0.75]
    assert result.calibration["win_rate"].to_list() == [0, 1]


def test_bin_edges_and_endpoints() -> None:
    result = probability_metrics([1, 0, 1, 0], [0, 0.5, 0.5, 1], bins=2)
    assert result.calibration["count"].to_list() == [1, 3]
    assert result.ece == pytest.approx(0.5)
    assert result.log_loss > 10 and result.log_loss < 100
    perfect = probability_metrics([0, 1], [0, 1])
    assert perfect.ece == 0 and perfect.brier_score == 0
    assert perfect.calibration["mean_probability"].null_count() == 8


def test_weighted_metrics_and_zero_weight_bins() -> None:
    result = probability_metrics(
        [0, 1, 0], [0.2, 0.8, 0.8], bins=2, sample_weight=[0.5, 0.5, 1]
    )
    assert result.ece == pytest.approx(0.4)
    assert result.brier_score == pytest.approx(0.34)
    assert result.log_loss == pytest.approx((-log(0.8) - log(0.2)) / 2)
    ignored = probability_metrics([0, 1], [0.1, 0.9], bins=2, sample_weight=[0, 1])
    assert ignored.ece == pytest.approx(0.1)
    assert ignored.calibration["win_rate"][0] is None


def test_plot_can_export_without_display(tmp_path: Path) -> None:
    result = probability_metrics([0, 1, 1], [0.1, 0.5, 0.9])
    figure = plot_calibration(result)
    figure.savefig(tmp_path / "calibration.png")
    assert (tmp_path / "calibration.png").stat().st_size > 1000
    assert len(figure.axes) == 2
    curve = figure.axes[0].lines[1]
    assert np.asarray(curve.get_xdata()).tolist() == [0.1, 0.5, 0.9]
    assert np.asarray(curve.get_ydata()).tolist() == [0, 1, 1]


@pytest.mark.parametrize(
    "labels,probabilities",
    [
        ([], []),
        ([0], []),
        ([2], [0.5]),
        ([0], [float("nan")]),
        ([0], [-0.1]),
        ([0], [1.1]),
        ([[0]], [[0.5]]),
    ],
)
def test_bad_predictions(labels: list, probabilities: list) -> None:
    with pytest.raises(ValueError):
        probability_metrics(labels, probabilities)


@pytest.mark.parametrize("weight", [[], [-1], [0], [float("nan")], [float("inf")]])
def test_bad_weights(weight: list[float]) -> None:
    with pytest.raises(ValueError, match="weights"):
        probability_metrics([0], [0.2], sample_weight=weight)


@pytest.mark.parametrize("bins", [0, -1, True, 1.5])
def test_bad_bin_count(bins: int) -> None:
    with pytest.raises(ValueError, match="bins"):
        probability_metrics([0], [0.2], bins=bins)


def test_weight_scale_does_not_overflow_log_loss() -> None:
    result = probability_metrics([1, 0], [0, 1], sample_weight=[1e308, 1])
    unweighted = probability_metrics([1, 0], [0, 1])
    assert result.log_loss == pytest.approx(unweighted.log_loss)
