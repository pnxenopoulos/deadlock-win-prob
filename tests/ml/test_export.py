"""ONNX portability, dynamic shapes, encoder contracts, and release integrity."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import polars as pl
import pytest
import torch

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from dwp.cli.export import main
from dwp.graphs.snapshots import GraphSnapshot
from dwp.ml.checkpoints import save_model
from dwp.ml.export import export_onnx, onnx_inputs
from dwp.ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, GraphEncoder
from dwp.ml.model import HeterogeneousGraphTransformer


@pytest.fixture
def exported(tmp_path: Path, graph_snapshot: GraphSnapshot):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    encoder = GraphEncoder.fit([graph_snapshot])
    model = HeterogeneousGraphTransformer(
        encoder, hidden_dim=16, num_layers=1, dropout=0.2
    ).eval()
    checkpoint = tmp_path / "best.pt"
    save_model(checkpoint, model, encoder)
    output = tmp_path / "bundle"
    export_onnx(checkpoint, output, model_version="test-version")
    yield model, encoder, output
    torch.set_num_threads(previous)


def test_onnx_matches_native_with_dynamic_nodes_tokens_padding_and_empty_types(
    exported, graph_snapshot
):
    model, encoder, output = exported
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(output / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    empty = replace(
        graph_snapshot,
        players=graph_snapshot.players.clear(),
        objectives=graph_snapshot.objectives.clear(),
        ziplines=graph_snapshot.ziplines.clear(),
    )
    inventories = replace(
        graph_snapshot,
        players=graph_snapshot.players.with_columns(
            pl.Series("upgrade_ids", [[1001] * 9, None])
        ),
    )
    for snapshots in ([graph_snapshot], [empty], [inventories, graph_snapshot, empty]):
        graphs = [encoder.transform(g) for g in snapshots]
        with torch.inference_mode():
            expected = model(graphs).sigmoid().numpy()
        inputs = onnx_inputs(graphs)
        assert inputs["player_numeric"].shape[1] >= 1
        assert inputs["upgrades"].shape[-1] >= 1
        actual = session.run(None, inputs)[0]
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    # Alter all padded node features; masks must keep predictions unchanged.
    inputs = onnx_inputs([encoder.transform(empty)])
    expected = session.run(None, inputs)[0]
    for name in ("player_numeric", "objective_numeric", "zipline_numeric"):
        inputs[name].fill(10000)
    inputs["teams"][:, 1:] = 4
    inputs["positions"][:, 1:] = 10000
    np.testing.assert_allclose(
        session.run(None, inputs)[0], expected, rtol=2e-5, atol=2e-5
    )


def test_export_metadata_and_checksums_are_self_contained(exported):
    model, encoder, output = exported
    manifest = json.loads((output / "manifest.json").read_text())
    metadata = json.loads((output / "encoder.json").read_text())
    assert manifest["model_version"] == "test-version"
    assert manifest["model_config"] == model.config
    assert manifest["output"]["label_1"] == "Archmother"
    assert metadata["numeric_features"] == [list(names) for names in NUMERIC_FEATURES]
    assert metadata["categorical_features"] == [
        list(names) for names in CATEGORICAL_FEATURES
    ]
    assert metadata["vocabularies"] == encoder.vocabularies
    assert metadata["means"] == [v.tolist() for v in encoder.means]
    assert metadata["scales"] == [v.tolist() for v in encoder.scales]
    assert metadata["unknown_team_index"] == 0
    assert metadata["team_indices"] == [
        {"raw": None, "index": 0},
        {"raw": 0, "index": 1},
        {"raw": 1, "index": 2},
        {"raw": 2, "index": 3},
        {"raw": 3, "index": 4},
    ]
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with ZipFile(output / "win-probability-model.zip") as bundle:
        assert "model.onnx" in bundle.namelist() and "encoder.json" in bundle.namelist()
        assert all("/" not in name for name in bundle.namelist())
        assert bundle.read("model.pt") == (output / "model.pt").read_bytes()
    with pytest.raises(FileExistsError):
        export_onnx(output / "model.pt", output)


def test_export_cli_uses_checkpoint_and_records_provenance(tmp_path, graph_snapshot):
    encoder = GraphEncoder.fit([graph_snapshot])
    model = HeterogeneousGraphTransformer(encoder, hidden_dim=16, num_layers=1).eval()
    checkpoint = tmp_path / "model.pt"
    save_model(checkpoint, model, encoder)
    (tmp_path / "metrics.json").write_text('{"best_epoch":2}')
    (tmp_path / "run.json").write_text(
        '{"data":{"preparation_plan":{"data_version":null}}}'
    )
    main(
        [
            str(checkpoint),
            "--output",
            str(tmp_path / "release"),
            "--model-version",
            "model-ci-test",
        ]
    )
    assert json.loads((tmp_path / "release" / "metrics.json").read_text()) == {
        "best_epoch": 2
    }
    assert (tmp_path / "release" / "run.json").read_bytes() == (
        tmp_path / "run.json"
    ).read_bytes()


def test_failed_parity_never_produces_a_release_manifest(
    tmp_path, graph_snapshot, monkeypatch
):
    encoder = GraphEncoder.fit([graph_snapshot])
    model = HeterogeneousGraphTransformer(encoder, hidden_dim=16, num_layers=1).eval()
    checkpoint = tmp_path / "model.pt"
    save_model(checkpoint, model, encoder)
    monkeypatch.setattr(
        ort.InferenceSession,
        "run",
        lambda self, *args, **kwargs: [np.array([np.nan], dtype=np.float32)],
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        export_onnx(checkpoint, tmp_path / "failed")
    assert not (tmp_path / "failed" / "manifest.json").exists()
    assert not (tmp_path / "failed" / "win-probability-model.zip").exists()
