"""Padding, graph isolation, batched kernels, and gradient equivalence."""

from dataclasses import replace

import polars as pl
import pytest
import torch
from torch.nn import functional as F

from dwp.graphs.snapshots import GraphSnapshot
from dwp.ml.batching import GraphBatch
from dwp.ml.features import GraphEncoder
from dwp.ml.model import HeterogeneousGraphTransformer


@pytest.fixture(autouse=True, scope="module")
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def varied_graphs(graph_snapshot: GraphSnapshot):
    snapshots = [
        graph_snapshot,
        replace(
            graph_snapshot,
            players=graph_snapshot.players.head(1).with_columns(
                pl.Series("upgrade_ids", [[1001, 1002, 1001, 1002]])
            ),
            objectives=graph_snapshot.objectives.clear(),
            ziplines=graph_snapshot.ziplines.head(1),
        ),
        replace(
            graph_snapshot,
            players=graph_snapshot.players.clear(),
            objectives=graph_snapshot.objectives.clear(),
            ziplines=graph_snapshot.ziplines.clear(),
        ),
        replace(
            graph_snapshot,
            players=pl.concat([graph_snapshot.players, graph_snapshot.players.head(1)]),
            objectives=graph_snapshot.objectives.head(1),
            ziplines=graph_snapshot.ziplines.clear(),
        ),
    ]
    encoder = GraphEncoder.fit(snapshots)
    return encoder, [encoder.transform(graph) for graph in snapshots]


def test_collation_preserves_connections_and_node_presence(varied_graphs) -> None:
    _, graphs = varied_graphs
    batch = GraphBatch.from_graphs(graphs).to("cpu")
    assert batch.valid.sum(dim=1).tolist() == [7, 3, 1, 5]
    assert batch.upgrades.shape == (4, 3, 4)
    zipline_slots = (batch.node_types == 3).nonzero().flatten()
    assert batch.connections[0, zipline_slots[0], zipline_slots[1]]
    assert batch.connections.sum() == sum(graph.connections.sum() for graph in graphs)
    assert batch.connections_known[0, zipline_slots].tolist() == [True, False]
    assert not batch.position_known[~batch.valid].any()
    with pytest.raises(ValueError, match="at least one graph"):
        GraphBatch.from_graphs([])


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_batched_predictions_and_gradients_match_individual_graphs(
    varied_graphs, device: str
) -> None:
    torch.manual_seed(17)
    encoder, graphs = varied_graphs
    model = HeterogeneousGraphTransformer(
        encoder, hidden_dim=16, num_layers=2, dropout=0
    ).to(device)
    batch = GraphBatch.from_graphs(graphs).to(device)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
    logits = model(batch)
    F.binary_cross_entropy_with_logits(logits, labels).backward()
    gradients = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    individual = torch.cat([model(graph.to(device)) for graph in graphs])
    F.binary_cross_entropy_with_logits(individual, labels).backward()
    torch.testing.assert_close(logits, individual, atol=2e-6, rtol=2e-5)
    assert gradients
    for name, parameter in model.named_parameters():
        if name in gradients:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(
                parameter.grad, gradients[name], atol=3e-6, rtol=2e-4
            )


def test_padding_values_cannot_change_real_graph_predictions(varied_graphs) -> None:
    encoder, graphs = varied_graphs
    model = HeterogeneousGraphTransformer(encoder, hidden_dim=16, num_layers=2).eval()
    batch = GraphBatch.from_graphs(graphs)
    counts = [values.shape[1] for values in batch.numeric]
    type_masks = batch.valid.split(counts, dim=1)
    changed = replace(
        batch,
        numeric=tuple(
            values.masked_fill(~mask[:, :, None], 10000)
            for values, mask in zip(batch.numeric, type_masks)
        ),
        categorical=tuple(
            values.masked_fill(~mask[:, :, None], 1)
            for values, mask in zip(batch.categorical, type_masks)
        ),
        upgrades=batch.upgrades.masked_fill(~type_masks[1][:, :, None], 1),
        teams=batch.teams.masked_fill(~batch.valid, 4),
        positions=batch.positions.masked_fill(~batch.valid[:, :, None], 10000),
        position_known=batch.position_known | ~batch.valid,
        connections=batch.connections
        | ~(batch.valid[:, :, None] & batch.valid[:, None, :]),
        connections_known=batch.connections_known | ~batch.valid,
    )
    with torch.inference_mode():
        torch.testing.assert_close(model(changed), model(batch))


def test_attention_layer_runs_once_for_the_whole_batch(varied_graphs) -> None:
    encoder, graphs = varied_graphs
    model = HeterogeneousGraphTransformer(encoder, hidden_dim=16, num_layers=1).eval()
    shapes = []
    handle = model.get_submodule("blocks.0.qkv.1").register_forward_pre_hook(
        lambda module, inputs: shapes.append(tuple(inputs[0].shape))
    )
    with torch.inference_mode():
        model(graphs)
    handle.remove()
    assert shapes == [(4, 3, 16)]
