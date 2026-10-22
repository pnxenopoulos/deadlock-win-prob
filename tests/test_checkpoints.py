"""Checkpoint round trips preserve model settings, features, and predictions."""

from pathlib import Path

import pytest
import torch

from dwp.checkpoints import load_model, save_model
from dwp.features import GraphEncoder
from dwp.model import HeterogeneousGraphTransformer
from dwp.representations import GraphSnapshot


def test_portable_checkpoint_round_trip(
    tmp_path: Path, graph_snapshot: GraphSnapshot
) -> None:
    encoder = GraphEncoder.fit([graph_snapshot])
    model = HeterogeneousGraphTransformer(
        encoder, hidden_dim=16, num_layers=2, dropout=0.2
    ).eval()
    path = tmp_path / "model.pt"
    save_model(path, model, encoder)
    restored, features = load_model(path)
    assert restored.config == model.config and not restored.training
    assert features.vocabularies == encoder.vocabularies
    with torch.no_grad():
        torch.testing.assert_close(
            model(encoder.transform(graph_snapshot)),
            restored(features.transform(graph_snapshot)),
        )
    payload = torch.load(path, weights_only=True)
    payload["numeric_features"] = []
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema"):
        load_model(path)
