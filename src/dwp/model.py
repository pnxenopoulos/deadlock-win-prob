"""Small heterogeneous graph transformer for P(Archmother wins)."""

from collections.abc import Sequence
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# PyTorch declares forward as a mutable Callable; overriding it is its intended API.
from dwp.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    GraphEncoder,
    GraphTensors,
)


class ModelConfig(TypedDict):
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float


class _AttentionBlock(nn.Module):
    """Type-specific projections with pair-type/team/geometry attention biases."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim * 3) for _ in range(4)
        )
        self.output = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(4))
        self.relation_bias = nn.Embedding(4 * 4 * 5, heads)
        self.geometry_bias = nn.Sequential(
            nn.Linear(7, 32), nn.GELU(), nn.Linear(32, heads)
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(  # ty: ignore[invalid-mutable-override]
        self, x: Tensor, graph: GraphTensors, relations: Tensor, geometry: Tensor
    ) -> Tensor:
        counts = [t.shape[0] for t in graph.numeric]
        pieces = self.norm1(x).split(counts)
        projections = torch.cat([self.qkv[i](part) for i, part in enumerate(pieces)])
        q, k, v = (
            projections.reshape(x.shape[0], 3, self.heads, -1)
            .permute(1, 2, 0, 3)
            .unbind(0)
        )
        bias = self.relation_bias(relations) + self.geometry_bias(geometry)
        # Complete directed graph, excluding self edges; the residual retains self state.
        bias = bias.permute(2, 0, 1).masked_fill(
            torch.eye(x.shape[0], dtype=torch.bool, device=x.device), float("-inf")
        )
        attended = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=bias,
                dropout_p=self.dropout if self.training else 0.0,
            )
            .transpose(0, 1)
            .reshape_as(x)
        )
        projected = torch.cat(
            [self.output[i](part) for i, part in enumerate(attended.split(counts))]
        )
        x = x + F.dropout(projected, p=self.dropout, training=self.training)
        return x + self.feed_forward(self.norm2(x))


class HeterogeneousGraphTransformer(nn.Module):
    """Type-specific encoders and attention, with team/type-balanced readout.

    Forward returns one raw logit per graph (shape [batch]); use
    BCEWithLogitsLoss for training and logits.sigmoid() for probabilities.
    Graphs are processed separately, so padding never mixes matches. Dense
    all-pairs attention costs O(nodes**2); no Polars edge table is needed.
    """

    def __init__(
        self,
        encoder: GraphEncoder,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads or num_layers < 1:
            raise ValueError(
                "dimensions/layers must be positive and hidden_dim divisible by num_heads"
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.config: ModelConfig = {
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        self.encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(len(names) * 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            for names in NUMERIC_FEATURES
        )
        self.categories = nn.ModuleDict(
            {
                name: nn.Embedding(len(vocab) + 2, hidden_dim, padding_idx=0)
                for name, vocab in encoder.vocabularies.items()
                if name != "upgrade_ids"
            }
        )
        self.upgrades = nn.Embedding(
            len(encoder.vocabularies["upgrade_ids"]) + 2, hidden_dim, padding_idx=0
        )
        self.teams = nn.Embedding(5, hidden_dim)
        self.blocks = nn.ModuleList(
            _AttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(20 * (hidden_dim + 1), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _graph_logit(self, graph: GraphTensors) -> Tensor:
        pieces = []
        for i, numeric in enumerate(graph.numeric):
            x = self.encoders[i](numeric)
            for column, name in enumerate(CATEGORICAL_FEATURES[i]):
                x = x + self.categories[name](graph.categorical[i][:, column])
            if i == 1:
                x = x + self.upgrades(graph.upgrades).sum(dim=1) / (
                    graph.upgrades != 0
                ).sum(dim=1, keepdim=True).clamp_min(1)
            pieces.append(x)
        x = torch.cat(pieces) + self.teams(graph.teams)
        team_a, team_b = graph.teams[:, None], graph.teams[None, :]
        playing = (team_a >= 3) & (team_b >= 3)
        relations = torch.full_like(team_a + team_b, 4)  # Unknown.
        relations = torch.where((team_a == 1) | (team_b == 1), 3, relations)  # Neutral.
        relations = torch.where(playing & (team_a == team_b), 1, relations)
        relations = torch.where(playing & (team_a != team_b), 2, relations)
        global_pair = (graph.node_types[:, None] == 0) | (
            graph.node_types[None, :] == 0
        )
        relations = torch.where(global_pair, 0, relations)
        relations = (
            graph.node_types[:, None] * 4 + graph.node_types[None, :]
        ) * 5 + relations
        known = graph.position_known[:, None] & graph.position_known[None, :]
        delta = (graph.positions[:, None] - graph.positions[None, :]) * known[
            :, :, None
        ]
        geometry = torch.cat(
            [
                delta,
                delta.norm(dim=-1, keepdim=True),
                known[:, :, None].float(),
                graph.connections[:, :, None].float(),
                (graph.connections_known[:, None] & (graph.node_types[None, :] == 3))[
                    :, :, None
                ].float(),
            ],
            dim=-1,
        )
        for block in self.blocks:
            x = block(x, graph, relations, geometry)
        # Twenty independent buckets (four node types x five team states).
        bucket = graph.node_types * 5 + graph.teams
        count = torch.bincount(bucket, minlength=20).to(x.dtype)
        pooled = x.new_zeros((20, x.shape[-1])).index_add(0, bucket, x) / count[
            :, None
        ].clamp_min(1)
        summary = torch.cat([pooled.flatten(), count.log1p()])
        return self.readout(summary).squeeze(-1)

    def forward(self, graphs: GraphTensors | Sequence[GraphTensors]) -> Tensor:  # ty: ignore[invalid-mutable-override]
        """Return [batch] logits; call model.eval() before evaluating probabilities."""
        batch = [graphs] if isinstance(graphs, GraphTensors) else graphs
        if not batch:
            raise ValueError("at least one graph is required")
        return torch.stack([self._graph_logit(graph) for graph in batch])
