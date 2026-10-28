"""Export checkpoint weights and a versioned encoder contract for ONNX clients."""

import hashlib
import json
import shutil
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from dwp.ml.batching import GraphBatch
from dwp.ml.checkpoints import load_model
from dwp.ml.features import (
    _TEAMS,
    CATEGORICAL_FEATURES,
    NODE_TYPES,
    NUMERIC_FEATURES,
    GraphTensors,
)
from dwp.ml.model import HeterogeneousGraphTransformer

_INPUTS = (
    "global_numeric",
    "player_numeric",
    "objective_numeric",
    "zipline_numeric",
    "player_categorical",
    "objective_categorical",
    "zipline_categorical",
    "upgrades",
    "teams",
    "positions",
    "position_known",
    "connections",
    "connections_known",
    "valid",
)


class _OnnxModel(nn.Module):
    def __init__(self, model: HeterogeneousGraphTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, values: tuple[Tensor, ...]) -> Tensor:  # ty: ignore[invalid-mutable-override]
        numeric = values[:4]
        batch = numeric[0].shape[0]
        graph = GraphBatch(
            numeric=numeric,
            categorical=(values[4].new_empty((batch, 1, 0)), *values[4:7]),
            upgrades=values[7],
            node_types=torch.cat(
                [values[8].new_full((t.shape[1],), i) for i, t in enumerate(numeric)]
            ),
            teams=values[8],
            positions=values[9],
            position_known=values[10],
            connections=values[11],
            connections_known=values[12],
            valid=values[13],
        )
        return self.model(graph).sigmoid()


def onnx_inputs(graphs: Sequence[GraphTensors]) -> dict[str, np.ndarray]:
    """Collate CPU features to the browser contract, masking empty type slots."""
    batch = GraphBatch.from_graphs(graphs, pad_empty_types=True).to("cpu")
    values = (
        *batch.numeric,
        *batch.categorical[1:],
        batch.upgrades,
        batch.teams,
        batch.positions,
        batch.position_known,
        batch.connections,
        batch.connections_known,
        batch.valid,
    )
    return {name: tensor.numpy() for name, tensor in zip(_INPUTS, values, strict=True)}


def _probe(
    counts: tuple[int, int, int, int],
    batch: int,
    tokens: int,
    vocabularies: dict[str, dict[str, int]],
    seed: int,
) -> tuple[Tensor, ...]:
    """Deterministic, label-free probes, including an isolated global-only graph."""
    rng = torch.Generator().manual_seed(seed)
    nodes = sum(counts)
    numeric = [
        torch.randn(batch, n, len(columns) * 2, generator=rng)
        for n, columns in zip(counts, NUMERIC_FEATURES)
    ]
    for values in numeric:
        values[:, :, values.shape[-1] // 2 :] = (
            torch.rand(
                values.shape[0], values.shape[1], values.shape[-1] // 2, generator=rng
            )
            > 0.3
        ).float()
    categorical = [
        torch.stack(
            [
                torch.randint(
                    len(vocabularies[name]) + 2, (batch, counts[i]), generator=rng
                )
                for name in CATEGORICAL_FEATURES[i]
            ],
            dim=-1,
        )
        for i in range(1, 4)
    ]
    upgrades = torch.randint(
        len(vocabularies["upgrade_ids"]) + 2, (batch, counts[1], tokens), generator=rng
    )
    teams = torch.randint(5, (batch, nodes), generator=rng)
    positions = torch.randn(batch, nodes, 3, generator=rng)
    known = torch.rand(batch, nodes, generator=rng) > 0.3
    connections = torch.rand(batch, nodes, nodes, generator=rng) > 0.9
    connections_known = torch.rand(batch, nodes, generator=rng) > 0.3
    valid = torch.ones(batch, nodes, dtype=torch.bool)
    if seed == 1:
        valid[0, 1:] = False
    # Mix partial padding with fully populated entries when batched.
    if batch > 1:
        valid[1, -1] = False
    return (
        *numeric,
        *categorical,
        upgrades,
        teams,
        positions,
        known,
        connections,
        connections_known,
        valid,
    )


def export_onnx(
    checkpoint: Path, output: Path, *, model_version: str | None = None
) -> Path:
    """Export a CPU checkpoint, validate parity, and package its encoder metadata.

    Requires the optional export dependencies. Output must not exist. The bundle
    accepts encoded tensors with dynamic batch/node/token dimensions, at least
    one slot per node type and one token slot; valid=false masks placeholder nodes.
    Returns manifest.json only after ONNX checking and runtime parity pass.
    """
    import onnx
    import onnxruntime as ort

    model, encoder = load_model(checkpoint)
    source_run: dict[str, Any] = (
        json.loads(checkpoint.with_name("run.json").read_text(encoding="utf-8"))
        if checkpoint.with_name("run.json").is_file()
        else {}
    )
    wrapper = _OnnxModel(model).eval()
    output.mkdir(parents=True, exist_ok=False)
    example = _probe((1, 3, 2, 4), 2, 3, encoder.vocabularies, 0)
    dim = torch.export.Dim
    b, p, o, z, t = (
        dim(name, min=1)
        for name in ("batch", "players", "objectives", "ziplines", "tokens")
    )
    n = dim.AUTO
    shapes = (
        {0: b},
        {0: b, 1: p},
        {0: b, 1: o},
        {0: b, 1: z},
        {0: b, 1: p},
        {0: b, 1: o},
        {0: b, 1: z},
        {0: b, 1: p, 2: t},
        {0: b, 1: n},
        {0: b, 1: n},
        {0: b, 1: n},
        {0: b, 1: n, 2: n},
        {0: b, 1: n},
        {0: b, 1: n},
    )
    with sdpa_kernel(SDPBackend.MATH):
        torch.onnx.export(
            wrapper,
            (example,),
            output / "model.onnx",
            input_names=list(_INPUTS),
            output_names=["archmother_win_probability"],
            dynamic_shapes=(shapes,),
            dynamo=True,
            opset_version=18,
            external_data=False,
        )
    onnx.checker.check_model(str(output / "model.onnx"), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(output / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    errors = []
    with torch.inference_mode():
        for seed, (counts, batch, tokens) in enumerate(
            (
                ((1, 1, 1, 1), 1, 1),
                ((1, 2, 3, 5), 3, 7),
                ((1, 12, 24, 334), 1, 32),
            ),
            1,
        ):
            values = _probe(counts, batch, tokens, encoder.vocabularies, seed)
            expected = wrapper(values).numpy()
            actual = np.asarray(
                session.run(
                    None,
                    {name: v.numpy() for name, v in zip(_INPUTS, values, strict=True)},
                )[0],
                dtype=np.float32,
            )
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise RuntimeError("non-finite ONNX parity predictions")
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            errors.append(float(np.max(np.abs(actual - expected))))
    metadata = {
        "format_version": 1,
        "node_type_order": list(NODE_TYPES),
        "numeric_features": [list(names) for names in NUMERIC_FEATURES],
        "categorical_features": [list(names) for names in CATEGORICAL_FEATURES],
        "means": [v.tolist() for v in encoder.means],
        "scales": [v.tolist() for v in encoder.scales],
        "vocabularies": encoder.vocabularies,
        "team_indices": [{"raw": raw, "index": index} for raw, index in _TEAMS.items()],
        "unknown_team_index": 0,
        "numeric_encoding": {
            "transform": "sign(x) * log1p(abs(x))",
            "computation": "float64 preprocessing, then cast values and masks to float32",
            "standardization": "(transformed - mean) / scale",
            "missing_or_nonfinite_value": 0,
            "layout": "standardized features followed by observed masks in numeric_features order",
        },
        "categorical_encoding": {
            "missing_or_padding": 0,
            "unseen": 1,
            "vocabulary_keys": "Python string form of recorded values",
        },
        "positions": {
            "axes": ["x", "y", "z"],
            "scale": 10000,
            "known": "all three raw coordinates finite; otherwise position_known=false and position=[0,0,0]",
        },
        "upgrades": {
            "source": "recorded upgrade_ids, including items and abilities",
            "padding": 0,
            "duplicates": "preserve",
            "pooling": "mean of non-padding token embeddings",
        },
        "derived_numeric": {
            "health_fraction": "health / max_health when max_health > 0; otherwise missing",
            "upgrade_count": "length of upgrade_ids; null list remains missing",
            "item_count": "length of item_ids when catalog enrichment is enabled; otherwise missing",
            "baseline_features": "matching baseline_stats value by stat name; otherwise missing",
        },
        "sampling": {
            "paused_ticks": "excluded",
            "elapsed_seconds": "active recording time at this tick; not reconstructed match time",
            "future_state": "excluded; state uses backward as-of observations",
        },
        "connections": {
            "source": "directed zipline connections resolved by full node handles",
            "known": "connections list is not null; [] is known empty",
        },
        "global_node": "exactly one valid global node per graph, first in concatenated node order",
        "padding": {
            "minimum_nodes_per_type": 1,
            "minimum_upgrade_tokens": 1,
            "invalid_nodes": "valid=false; zero numeric, categorical, tokens, teams, positions and connection fields",
        },
    }
    (output / "encoder.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    shutil.copyfile(checkpoint, output / "model.pt")
    for name in ("run.json", "metrics.json", "summary.md"):
        if checkpoint.with_name(name).is_file():
            shutil.copyfile(checkpoint.with_name(name), output / name)
    parity = {
        "runtime": "onnxruntime CPUExecutionProvider",
        "absolute_tolerance": 2e-5,
        "relative_tolerance": 2e-5,
        "max_absolute_probability_error": max(errors),
        "probe_cases": len(errors),
    }
    (output / "parity.json").write_text(
        json.dumps(parity, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "format_version": 1,
        "model_version": model_version
        or hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model": "model.onnx",
        "encoder": "encoder.json",
        "checkpoint": "model.pt",
        "opset": 18,
        "model_config": model.config,
        "data_version": source_run.get("data", {})
        .get("preparation_plan", {})
        .get("data_version", source_run.get("data", {}).get("data_version")),
        "dimensions": {
            "batch": "number of graphs, at least 1",
            "players": "padded player slots, at least 1",
            "objectives": "padded objective slots, at least 1",
            "ziplines": "padded zipline slots, at least 1",
            "tokens": "padded upgrade token slots per player, at least 1",
            "nodes": "1 + players + objectives + ziplines; every node-axis input must use this same size",
        },
        "output": {
            "name": "archmother_win_probability",
            "dtype": "float32",
            "shape": ["batch"],
            "label_1": "Archmother",
            "label_0": "Hidden King",
            "hidden_king_probability": "1 - archmother_win_probability",
        },
        "inputs": [
            {"name": i.name, "dtype": i.type, "shape": i.shape}
            for i in session.get_inputs()
        ],
        "versions": {
            name: version(name)
            for name in (
                "deadlock-win-prob",
                "boon-deadlock",
                "torch",
                "onnx",
                "onnxruntime",
                "onnxscript",
            )
        },
        "validation": parity,
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir())
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
            for p in sorted(output.iterdir())
        ),
        encoding="utf-8",
    )
    files = sorted(output.iterdir())
    with ZipFile(
        output / "win-probability-model.zip", "w", compression=ZIP_DEFLATED
    ) as bundle:
        for path in files:
            bundle.write(path, path.name)
    return output / "manifest.json"
