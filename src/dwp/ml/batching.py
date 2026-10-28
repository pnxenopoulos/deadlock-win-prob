"""Collate encoded graphs into padded tensor batches without mixing matches."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import Self

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from dwp.ml.features import GraphTensors


@dataclass(frozen=True, slots=True)
class GraphBatch:
    """Batch-first features, padded separately for each node type.

    numeric/categorical contain four [batch, type_nodes, features] tensors;
    upgrades is [batch, players, tokens]. Other node fields use concatenated
    type slots. node_types is shared across graphs; valid marks real nodes.
    """

    numeric: tuple[Tensor, ...]
    categorical: tuple[Tensor, ...]
    upgrades: Tensor
    node_types: Tensor
    teams: Tensor
    positions: Tensor
    position_known: Tensor
    connections: Tensor
    connections_known: Tensor
    valid: Tensor

    @classmethod
    def from_graphs(
        cls, graphs: Sequence[GraphTensors], *, pad_empty_types: bool = False
    ) -> Self:
        """Pad one batch on the input device; feature encoding stays on demand."""
        if not graphs:
            raise ValueError("at least one graph is required")
        first = graphs[0]
        numeric = tuple(
            pad_sequence([g.numeric[i] for g in graphs], batch_first=True)
            for i in range(4)
        )
        categorical = tuple(
            pad_sequence([g.categorical[i] for g in graphs], batch_first=True)
            for i in range(4)
        )
        if pad_empty_types:
            numeric = tuple(F.pad(t, (0, 0, 0, int(t.shape[1] == 0))) for t in numeric)
            categorical = tuple(
                F.pad(t, (0, 0, 0, int(t.shape[1] == 0))) for t in categorical
            )
        counts = [values.shape[1] for values in numeric]
        offsets = list(accumulate(counts, initial=0))
        size, nodes = len(graphs), offsets[-1]
        tokens = max(g.upgrades.shape[1] for g in graphs)
        upgrades = torch.stack(
            [
                F.pad(
                    g.upgrades,
                    (
                        0,
                        tokens - g.upgrades.shape[1],
                        0,
                        counts[1] - g.upgrades.shape[0],
                    ),
                )
                for g in graphs
            ]
        )
        node_types = torch.cat(
            [first.node_types.new_full((count,), i) for i, count in enumerate(counts)]
        )
        teams = first.teams.new_zeros((size, nodes))
        positions = first.positions.new_zeros((size, nodes, 3))
        position_known = first.position_known.new_zeros((size, nodes))
        connections = first.connections.new_zeros((size, nodes, nodes))
        connections_known = first.connections_known.new_zeros((size, nodes))
        valid = first.position_known.new_zeros((size, nodes))
        for row, graph in enumerate(graphs):
            indices = torch.cat(
                [
                    torch.arange(values.shape[0], device=teams.device) + offsets[i]
                    for i, values in enumerate(graph.numeric)
                ]
            )
            valid[row, indices] = True
            teams[row, indices] = graph.teams
            positions[row, indices] = graph.positions
            position_known[row, indices] = graph.position_known
            connections_known[row, indices] = graph.connections_known
            connections[row, indices[:, None], indices[None, :]] = graph.connections
        return cls(
            numeric,
            categorical,
            upgrades,
            node_types,
            teams,
            positions,
            position_known,
            connections,
            connections_known,
            valid,
        )

    def to(self, device: torch.device | str) -> Self:
        """Transfer collated tensors together, including the padding mask."""
        return replace(
            self,
            numeric=tuple(t.to(device) for t in self.numeric),
            categorical=tuple(t.to(device) for t in self.categorical),
            upgrades=self.upgrades.to(device),
            node_types=self.node_types.to(device),
            teams=self.teams.to(device),
            positions=self.positions.to(device),
            position_known=self.position_known.to(device),
            connections=self.connections.to(device),
            connections_known=self.connections_known.to(device),
            valid=self.valid.to(device),
        )
