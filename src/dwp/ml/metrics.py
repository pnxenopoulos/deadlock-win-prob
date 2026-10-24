"""Binary win-probability metrics and reliability diagrams."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    """Scores plus all equal-width calibration bins, including empty bins."""

    log_loss: float
    ece: float
    brier_score: float
    calibration: pl.DataFrame


def probability_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
    sample_weight: Sequence[float] | None = None,
) -> ProbabilityMetrics:
    """Score P(Archmother wins) against binary outcomes.

    ECE is sum(bin weight * abs(mean probability - Archmother win rate)) /
    total weight: calibration of the positive event, not top-label confidence.
    Bins are [lower, upper), except the last includes 1. Endpoint probabilities
    are clipped only for log loss, using float64 machine epsilon. To weight
    matches equally, give each snapshot weight 1 / snapshots_in_its_match.
    """
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if (
        y.ndim != 1
        or p.shape != y.shape
        or not y.size
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("labels must be binary and probabilities finite in [0, 1]")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    w = (
        np.ones_like(p)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if (
        w.shape != y.shape
        or not np.isfinite(w).all()
        or (w < 0).any()
        or not 0 < w.sum() < np.inf
    ):
        raise ValueError("weights must be finite, nonnegative, and have positive sum")
    bin_id = np.minimum((p * bins).astype(int), bins - 1)
    count = np.bincount(bin_id, minlength=bins)
    weight = np.bincount(bin_id, weights=w, minlength=bins)
    nonempty = weight > 0
    mean_p = np.full(bins, np.nan)
    win_rate = np.full(bins, np.nan)
    mean_p[nonempty] = (
        np.bincount(bin_id, weights=w * p, minlength=bins)[nonempty] / weight[nonempty]
    )
    win_rate[nonempty] = (
        np.bincount(bin_id, weights=w * y, minlength=bins)[nonempty] / weight[nonempty]
    )
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(p, epsilon, 1 - epsilon)
    return ProbabilityMetrics(
        log_loss=float(
            np.average(
                -y * np.log(clipped) - (1 - y) * np.log1p(-clipped), weights=w / w.sum()
            )
        ),
        ece=float(
            np.sum(weight[nonempty] * np.abs(mean_p[nonempty] - win_rate[nonempty]))
            / w.sum()
        ),
        brier_score=float(np.average((p - y) ** 2, weights=w / w.sum())),
        calibration=pl.DataFrame(
            {
                "lower": np.arange(bins) / bins,
                "upper": np.arange(1, bins + 1) / bins,
                "count": count,
                "weight": weight,
                "mean_probability": mean_p,
                "win_rate": win_rate,
            },
            nan_to_null=True,
        ),
    )


def plot_calibration(
    result: ProbabilityMetrics, *, title: str = "Win probability calibration"
) -> Figure:
    """Return a headless-safe figure with reliability and bin-weight panels.

    Use figure.savefig(path) to export it. Empty/zero-weight bins are omitted
    from the reliability curve, and remain visible in the distribution panel.
    """
    figure = Figure(figsize=(6, 7), layout="constrained")
    reliability, distribution = figure.subplots(2, 1, height_ratios=[3, 1], sharex=True)
    table = result.calibration
    populated = table.filter(pl.col("weight") > 0)
    reliability.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    reliability.plot(
        populated["mean_probability"], populated["win_rate"], "o-", label="Model"
    )
    reliability.set(
        xlim=(0, 1), ylim=(0, 1), ylabel="Observed Archmother win rate", title=title
    )
    reliability.legend()
    reliability.text(
        0.04, 0.96, f"ECE = {result.ece:.3f}", va="top", transform=reliability.transAxes
    )
    distribution.bar(
        (table["lower"] + table["upper"]) / 2,
        table["weight"],
        width=(table["upper"] - table["lower"]) * 0.9,
    )
    distribution.set(xlabel="Predicted P(Archmother wins)", ylabel="Weight")
    return figure
