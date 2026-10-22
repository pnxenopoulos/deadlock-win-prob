"""Training orchestration, validation-only selection, CLI, and saved artifacts."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
import torch

from dwp.checkpoints import load_model
from dwp.features import GraphEncoder
from dwp.metrics import probability_metrics
from dwp.preparation import ReplayMatch
from dwp.representations import GraphSnapshot
from dwp.train import main
from dwp.training import TrainingConfig, evaluate, train_model


@pytest.fixture(autouse=True, scope="module")
def small_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def matches(graph_snapshot: GraphSnapshot) -> list[ReplayMatch]:
    return [
        ReplayMatch(
            mid,
            mid % 2,
            f"https://example.com/{mid}.dem.bz2",
            999,
            1000,
            [
                replace(
                    graph_snapshot,
                    match_id=mid,
                    tick=100 + t,
                    players=graph_snapshot.players.with_columns(
                        pl.lit(mid * 10).alias("hero_id")
                    ),
                )
                for t in range(mid % 3 + 1)
            ],
        )
        for mid in range(10, 16)
    ]


def test_training_selects_validation_checkpoint_and_exports_reports(
    matches: list[ReplayMatch],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TrainingConfig(
        epochs=2,
        batch_size=2,
        device="cpu",
        model={"hidden_dim": 16, "num_heads": 4, "num_layers": 1, "dropout": 0},
    )
    fitted_ids = set()
    original_fit = GraphEncoder.fit

    def fit(graphs):
        selected = list(graphs)
        fitted_ids.update(g.match_id for g in selected)
        return original_fit(selected)

    monkeypatch.setattr("dwp.training.GraphEncoder.fit", fit)
    calls = []

    def score(model, encoder, selected, **kwargs):
        calls.append({m.match_id for m in selected})
        metrics, predictions = evaluate(model, encoder, selected, **kwargs)
        # Force epoch 1 to be chosen; the final reports still use actual predictions.
        if len(calls) <= 2:
            metrics = replace(metrics, log_loss=[0.1, 0.9][len(calls) - 1])
        return metrics, predictions

    monkeypatch.setattr("dwp.training.evaluate", score)
    output = tmp_path / "run"
    result = train_model(matches, output, config)
    manifest = pl.read_csv(output / "matches.csv")
    groups = {
        name: set(manifest.filter(pl.col("split") == name)["match_id"])
        for name in ("train", "validation", "test")
    }
    assert fitted_ids == groups["train"]
    assert calls == [groups["validation"]] * 3 + [groups["test"]]
    assert (
        not groups["train"] & groups["validation"]
        and not groups["train"] & groups["test"]
    )
    assert json.loads((output / "metrics.json").read_text())["best_epoch"] == 1
    assert len(json.loads((output / "history.json").read_text())) == 2
    assert "| test |" in (output / "summary.md").read_text()
    model, encoder = load_model(output / "model.pt")
    held_out = [m for m in matches if m.match_id in groups["test"]]
    actual, predictions = evaluate(model, encoder, held_out)
    assert actual.log_loss == pytest.approx(result["test"].log_loss)
    saved = pl.read_csv(output / "test_predictions.csv")
    assert saved["probability"].to_list() == predictions["probability"].to_list()
    for phase in ("test", "validation"):
        assert (output / f"{phase}_calibration.png").stat().st_size > 1000
        assert pl.read_csv(output / f"{phase}_calibration.csv").height == 10
    assert "versions" in json.loads((output / "run.json").read_text())


def test_evaluation_weights_each_match_equally(matches: list[ReplayMatch]) -> None:
    from dwp.model import HeterogeneousGraphTransformer

    encoder = GraphEncoder.fit(g for m in matches for g in m.graphs)
    model = HeterogeneousGraphTransformer(encoder, hidden_dim=16, num_layers=1)
    metrics, predictions = evaluate(model, encoder, matches)
    assert predictions.group_by("match_id").agg(pl.col("weight").sum())[
        "weight"
    ].to_list() == pytest.approx([1] * len(matches))
    expected = probability_metrics(
        predictions["label"].to_list(),
        predictions["probability"].to_list(),
        sample_weight=predictions["weight"].to_list(),
    )
    assert metrics.log_loss == expected.log_loss and not model.training


def test_cli_runs_the_pipeline(
    matches: list[ReplayMatch],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare = Mock(return_value=matches)
    monkeypatch.setattr("dwp.train.prepare_matches", prepare)
    output = tmp_path / "cli-run"
    main(
        [
            "--days",
            "2",
            "--matches",
            "6",
            "--epochs",
            "1",
            "--every",
            "2048",
            "--max-snapshots",
            "2",
            "--hidden-dim",
            "16",
            "--num-layers",
            "1",
            "--threads",
            "1",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )
    assert prepare.call_args.kwargs["days"] == 2
    assert prepare.call_args.kwargs["match_modes"] == ["Ranked"]
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((output / "metrics.json").read_text())
    assert printed["test_log_loss"] == saved["test"]["log_loss"]
    assert printed["test_ece"] == saved["test"]["ece"]
    assert json.loads((output / "run.json").read_text())["data"]["every"] == 2048


@pytest.mark.parametrize(
    "options",
    [
        ["--matches", "2"],
        ["--epochs", "0"],
        ["--threads", "0"],
        ["--learning-rate", "nan"],
        ["--hidden-dim", "15"],
        ["--match-modes", "bogus"],
        ["--candidate-limit", "1"],
    ],
)
def test_cli_rejects_invalid_options_before_network(
    options: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = Mock()
    monkeypatch.setattr("dwp.train.prepare_matches", prepare)
    with pytest.raises(SystemExit) as error:
        main(["--output", str(tmp_path / "new"), *options])
    assert error.value.code == 2
    prepare.assert_not_called()


def test_invalid_data_and_existing_output_are_rejected(
    matches: list[ReplayMatch], tmp_path: Path
) -> None:
    config = TrainingConfig(epochs=1)
    with pytest.raises(ValueError, match="duplicate"):
        train_model(matches + matches[:1], tmp_path / "new", config)
    future = replace(matches[0], game_over_tick=100)
    with pytest.raises(ValueError, match="precede"):
        train_model([future, *matches[1:]], tmp_path / "new", config)
    with pytest.raises(FileExistsError):
        train_model(matches, tmp_path, config)
