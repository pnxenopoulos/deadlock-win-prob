"""Portable model checkpoints, including training-fitted feature encoding."""

from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import torch
from torch import Tensor

from dwp.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, GraphEncoder
from dwp.model import HeterogeneousGraphTransformer, ModelConfig


class _Checkpoint(TypedDict):
    format_version: int
    state_dict: dict[str, Tensor]
    model_config: ModelConfig
    numeric_features: list[list[str]]
    categorical_features: list[list[str]]
    means: list[list[float]]
    scales: list[list[float]]
    vocabularies: dict[str, dict[str, int]]


def save_model(
    path: Path, model: HeterogeneousGraphTransformer, encoder: GraphEncoder
) -> None:
    """Save weights plus primitive encoder/config data, without pickled objects."""
    payload: _Checkpoint = {
        "format_version": 1,
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": model.config,
        "numeric_features": [list(names) for names in NUMERIC_FEATURES],
        "categorical_features": [list(names) for names in CATEGORICAL_FEATURES],
        "means": [values.tolist() for values in encoder.means],
        "scales": [values.tolist() for values in encoder.scales],
        "vocabularies": encoder.vocabularies,
    }
    torch.save(payload, path)


def load_model(
    path: Path, *, device: str = "cpu"
) -> tuple[HeterogeneousGraphTransformer, GraphEncoder]:
    """Restore an inference-ready model and encoder with weights_only loading."""
    payload = cast(_Checkpoint, torch.load(path, map_location="cpu", weights_only=True))
    if (
        payload["format_version"] != 1
        or payload["numeric_features"] != [list(names) for names in NUMERIC_FEATURES]
        or payload["categorical_features"]
        != [list(names) for names in CATEGORICAL_FEATURES]
    ):
        raise ValueError(
            "checkpoint format or feature schema does not match this package"
        )
    encoder = GraphEncoder(
        tuple(np.asarray(values, dtype=np.float64) for values in payload["means"]),
        tuple(np.asarray(values, dtype=np.float64) for values in payload["scales"]),
        payload["vocabularies"],
    )
    model = HeterogeneousGraphTransformer(encoder, **payload["model_config"])
    model.load_state_dict(cast(dict[str, Any], payload["state_dict"]))
    return model.to(device).eval(), encoder
