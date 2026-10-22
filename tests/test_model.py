"""Graph permutation invariance, batch isolation, and actual gradient learning."""

from dataclasses import replace

import polars as pl
import pytest
import torch

from dwp.features import GraphEncoder
from dwp.model import HeterogeneousGraphTransformer
from dwp.representations import GraphSnapshot


@pytest.fixture(autouse=True, scope="module")
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_default_architecture_and_backpropagation(
    graph_snapshot: GraphSnapshot,
) -> None:
    torch.manual_seed(0)
    encoder = GraphEncoder.fit([graph_snapshot])
    graph = encoder.transform(graph_snapshot)
    model = HeterogeneousGraphTransformer(encoder)
    assert len(model.blocks) == 3 and model.blocks[0].heads == 4
    logits = model([graph, graph])
    assert logits.shape == (2,) and torch.isfinite(logits).all()
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([1.0, 0.0])
    ).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    relation_grad = dict(model.named_parameters())["blocks.0.relation_bias.weight"].grad
    assert relation_grad is not None and relation_grad.abs().sum() > 0
    upgrade_grad = model.upgrades.weight.grad
    assert upgrade_grad is not None and upgrade_grad.abs().sum() > 0


def test_permutation_invariance_and_batch_isolation(
    graph_snapshot: GraphSnapshot,
) -> None:
    torch.manual_seed(1)
    encoder = GraphEncoder.fit([graph_snapshot])
    model = HeterogeneousGraphTransformer(encoder, hidden_dim=16, num_layers=2).eval()
    reordered = replace(
        graph_snapshot,
        players=graph_snapshot.players.reverse().with_columns(
            pl.col("upgrade_ids").list.reverse()
        ),
        objectives=graph_snapshot.objectives.reverse(),
        ziplines=graph_snapshot.ziplines.reverse(),
    )
    original, permuted = encoder.transform(graph_snapshot), encoder.transform(reordered)
    with torch.no_grad():
        expected = model(original)
        torch.testing.assert_close(model(permuted), expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            model([original, permuted]), expected.repeat(2), atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(model(original), expected)
        assert ((expected.sigmoid() >= 0) & (expected.sigmoid() <= 1)).all()


def test_global_only_graph_is_finite(graph_snapshot: GraphSnapshot) -> None:
    empty = replace(
        graph_snapshot,
        players=graph_snapshot.players.clear(),
        objectives=graph_snapshot.objectives.clear(),
        ziplines=graph_snapshot.ziplines.clear(),
    )
    encoder = GraphEncoder.fit([empty])
    model = HeterogeneousGraphTransformer(encoder, hidden_dim=16, num_layers=1)
    output = model(encoder.transform(empty))
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_optimizer_can_learn_distinct_states(graph_snapshot: GraphSnapshot) -> None:
    torch.manual_seed(7)
    opposite = replace(
        graph_snapshot,
        players=graph_snapshot.players.with_columns(pl.Series("health", [50, 100])),
    )
    encoder = GraphEncoder.fit([graph_snapshot, opposite])
    graphs = [encoder.transform(g) for g in (graph_snapshot, opposite)]
    model = HeterogeneousGraphTransformer(
        encoder, hidden_dim=16, num_layers=1, dropout=0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    labels = torch.tensor([0.0, 1.0])
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(graphs), labels
        )
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 0.5


@pytest.mark.parametrize(
    "settings",
    [
        {"hidden_dim": 15},
        {"num_heads": 0},
        {"num_layers": 0},
        {"dropout": 1},
        {"dropout": -0.1},
    ],
)
def test_invalid_configuration(graph_snapshot: GraphSnapshot, settings: dict) -> None:
    with pytest.raises(ValueError):
        HeterogeneousGraphTransformer(GraphEncoder.fit([graph_snapshot]), **settings)


def test_reject_empty_batch(graph_snapshot: GraphSnapshot) -> None:
    model = HeterogeneousGraphTransformer(GraphEncoder.fit([graph_snapshot]))
    with pytest.raises(ValueError, match="graph"):
        model([])
