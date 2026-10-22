"""Train, select by validation loss, and export held-out probability reports."""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from random import Random
from typing import Literal

import polars as pl
import torch
from torch.nn import functional as F

from dwp.checkpoints import load_model, save_model
from dwp.datasets import split_matches
from dwp.features import GraphEncoder
from dwp.metrics import ProbabilityMetrics, probability_metrics
from dwp.model import HeterogeneousGraphTransformer, ModelConfig
from dwp.preparation import ReplayMatch
from dwp.reporting import write_reports, write_run_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 3e-4
    seed: int = 42
    bins: int = 10
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    device: Literal["auto", "cpu", "cuda"] = "auto"
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            hidden_dim=128, num_heads=4, num_layers=3, dropout=0.1
        )
    )

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.bins) < 1:
            raise ValueError("epochs, batch_size, and bins must be positive")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            not isfinite(self.train_fraction)
            or not isfinite(self.validation_fraction)
            or self.train_fraction <= 0
            or self.validation_fraction <= 0
            or self.train_fraction + self.validation_fraction >= 1
        ):
            raise ValueError("split fractions must be positive and sum to < 1")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        model = self.model
        if (
            model["hidden_dim"] < 1
            or model["num_heads"] < 1
            or model["num_layers"] < 1
            or model["hidden_dim"] % model["num_heads"]
            or not 0 <= model["dropout"] < 1
        ):
            raise ValueError("invalid model dimensions, layers, heads, or dropout")


@torch.inference_mode()
def evaluate(
    model: HeterogeneousGraphTransformer,
    encoder: GraphEncoder,
    matches: Sequence[ReplayMatch],
    *,
    bins: int = 10,
) -> tuple[ProbabilityMetrics, pl.DataFrame]:
    """Evaluate all retained snapshots, giving each match equal total weight.

    Leaves the model in eval mode. Features are encoded one graph at a time,
    avoiding cached quadratic adjacency tensors for the whole dataset.
    """
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for match in matches:
        for graph in match.graphs:
            probability = model(encoder.transform(graph).to(device)).sigmoid().item()
            rows.append(
                {
                    "match_id": match.match_id,
                    "tick": graph.tick,
                    "label": match.label,
                    "probability": probability,
                    "weight": 1 / len(match.graphs),
                }
            )
    if not rows:
        raise ValueError("evaluation requires nonempty matches and snapshots")
    predictions = pl.DataFrame(rows)
    result = probability_metrics(
        predictions["label"].to_list(),
        predictions["probability"].to_list(),
        bins=bins,
        sample_weight=predictions["weight"].to_list(),
    )
    return result, predictions


def train_model(
    matches: Sequence[ReplayMatch],
    output: Path,
    config: TrainingConfig,
    *,
    data_config: Mapping[str, object] | None = None,
) -> dict[str, ProbabilityMetrics]:
    """Train with one random snapshot per training match per epoch.

    Split whole matches (80/10/10 by default), fit features on training only,
    select weights by match-weighted validation log loss, and score the test
    set once after selection. Output must not already exist. Saves portable
    weights/encoder/config plus provenance, history, predictions and plots.
    """
    if len({m.match_id for m in matches}) != len(matches):
        raise ValueError("duplicate match IDs")
    for match in matches:
        if match.label not in (0, 1) or not match.graphs:
            raise ValueError("every match needs a binary label and snapshots")
        if any(
            g.match_id != match.match_id or g.tick >= match.game_over_tick
            for g in match.graphs
        ):
            raise ValueError(
                "graph match IDs must agree and ticks must precede game over"
            )
    rows = pl.DataFrame({"match_id": [m.match_id for m in matches]})
    splits = split_matches(
        rows, config.train_fraction, config.validation_fraction, seed=config.seed
    )
    by_id = {match.match_id: match for match in matches}
    groups = {
        "train": [by_id[mid] for mid in splits.train["match_id"]],
        "validation": [by_id[mid] for mid in splits.validation["match_id"]],
        "test": [by_id[mid] for mid in splits.test["match_id"]],
    }
    device = (
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else config.device
    )
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    output.mkdir(parents=True, exist_ok=False)
    write_run_metadata(
        output, asdict(config), groups, device=device, data_config=data_config
    )
    torch.manual_seed(config.seed)
    rng = Random(config.seed)
    encoder = GraphEncoder.fit(g for match in groups["train"] for g in match.graphs)
    model = HeterogeneousGraphTransformer(encoder, **config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, int | float]] = []
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.sample(groups["train"], len(groups["train"]))
        training_loss = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            features = [
                encoder.transform(rng.choice(m.graphs)).to(device) for m in batch
            ]
            labels = torch.tensor(
                [m.label for m in batch], dtype=torch.float32, device=device
            )
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(features), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            training_loss += loss.detach().item() * len(batch)
        validation, _ = evaluate(model, encoder, groups["validation"], bins=config.bins)
        history.append(
            {
                "epoch": epoch,
                "train_log_loss": training_loss / len(order),
                "validation_log_loss": validation.log_loss,
                "validation_ece": validation.ece,
            }
        )
        logger.info(
            "Epoch %d/%d train loss=%.5f validation loss=%.5f ECE=%.5f",
            epoch,
            config.epochs,
            training_loss / len(order),
            validation.log_loss,
            validation.ece,
        )
        (output / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if validation.log_loss < best_loss:
            best_loss, best_epoch = validation.log_loss, epoch
            save_model(output / "model.pt", model, encoder)
    del optimizer, model
    model, encoder = load_model(output / "model.pt", device=device)
    evaluations = {
        name: evaluate(model, encoder, groups[name], bins=config.bins)
        for name in ("validation", "test")
    }
    write_reports(output, evaluations, best_epoch=best_epoch)
    for name, (metrics, _) in evaluations.items():
        logger.info(
            "%s log loss=%.6f ECE=%.6f Brier=%.6f",
            name,
            metrics.log_loss,
            metrics.ece,
            metrics.brier_score,
        )
    return {name: result[0] for name, result in evaluations.items()}
