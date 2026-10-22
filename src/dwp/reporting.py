"""Write training provenance, probability reports, and CI-friendly summaries."""

import json
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

import polars as pl

from dwp.metrics import ProbabilityMetrics, plot_calibration
from dwp.preparation import ReplayMatch


def write_run_metadata(
    output: Path,
    training_config: Mapping[str, object],
    groups: Mapping[str, Sequence[ReplayMatch]],
    *,
    device: str,
    data_config: Mapping[str, object] | None = None,
) -> None:
    """Record settings, versions, replay provenance, and disjoint match splits."""
    (output / "run.json").write_text(
        json.dumps(
            {
                "training": dict(training_config),
                "data": dict(data_config or {}),
                "device": device,
                "versions": {
                    name: version(name)
                    for name in (
                        "deadlock-win-prob",
                        "boon-deadlock",
                        "torch",
                        "polars",
                    )
                },
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    pl.DataFrame(
        [
            {
                "match_id": m.match_id,
                "label": m.label,
                "split": name,
                "snapshots": len(m.graphs),
                "replay_url": m.replay_url,
                "build": m.build,
                "game_over_tick": m.game_over_tick,
                "first_tick": min(g.tick for g in m.graphs),
                "last_tick": max(g.tick for g in m.graphs),
            }
            for name, group in groups.items()
            for m in group
        ]
    ).write_csv(output / "matches.csv")


def write_reports(
    output: Path,
    evaluations: Mapping[str, tuple[ProbabilityMetrics, pl.DataFrame]],
    *,
    best_epoch: int,
) -> None:
    """Save JSON scores, per-snapshot predictions, calibration bins/plots, summary."""
    report: dict[str, object] = {
        "best_epoch": best_epoch,
        "weighting": "equal total weight per match",
    }
    summary = [
        "# Deadlock win probability training",
        "",
        f"Best epoch: {best_epoch} (selected by validation log loss).",
        "",
        "| Split | Matches | Snapshots | Log loss | ECE | Brier score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, (metrics, predictions) in evaluations.items():
        matches = predictions["match_id"].n_unique()
        report[name] = {
            "matches": matches,
            "snapshots": predictions.height,
            "log_loss": metrics.log_loss,
            "ece": metrics.ece,
            "brier_score": metrics.brier_score,
        }
        predictions.write_csv(output / f"{name}_predictions.csv")
        metrics.calibration.write_csv(output / f"{name}_calibration.csv")
        figure = plot_calibration(
            metrics, title=f"{name.title()} win probability calibration"
        )
        figure.savefig(output / f"{name}_calibration.png", dpi=150)
        summary.append(
            f"| {name} | {matches} | {predictions.height} | {metrics.log_loss:.6f} | {metrics.ece:.6f} | {metrics.brier_score:.6f} |"
        )
    summary += [
        "",
        "Scores weight matches equally. Label 1 = Archmother; 0 = Hidden King.",
        "Test matches were excluded from fitting and checkpoint selection.",
        "",
    ]
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
