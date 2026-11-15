"""Checkpoint round trips preserve model settings, features, and predictions."""

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
import torch

from dwp.graphs.snapshots import GraphSnapshot
from dwp.graphs.waves import WAVE_FEATURES
from dwp.ml.checkpoints import load_model, save_model
from dwp.ml.features import GraphEncoder
from dwp.ml.model import HeterogeneousGraphTransformer


@pytest.mark.parametrize("wave_pressure", [False, True])
def test_portable_checkpoint_round_trip(
    tmp_path: Path, graph_snapshot: GraphSnapshot, wave_pressure: bool
) -> None:
    graph_snapshot = replace(
        graph_snapshot,
        global_info=graph_snapshot.global_info.with_columns(
            pl.lit(i).alias(name) for i, name in enumerate(WAVE_FEATURES)
        ),
    )
    encoder = GraphEncoder.fit([graph_snapshot], wave_pressure=wave_pressure)
    model = HeterogeneousGraphTransformer(
        encoder, hidden_dim=16, num_layers=2, dropout=0.2
    ).eval()
    path = tmp_path / "model.pt"
    save_model(path, model, encoder)
    restored, features = load_model(path)
    assert restored.config == model.config and not restored.training
    assert features.vocabularies == encoder.vocabularies
    assert features.wave_pressure == wave_pressure
    with torch.no_grad():
        torch.testing.assert_close(
            model(encoder.transform(graph_snapshot)),
            restored(features.transform(graph_snapshot)),
        )
    payload = torch.load(path, weights_only=True)
    payload["numeric_features"][3].append("connection_count")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema"):
        load_model(path)
