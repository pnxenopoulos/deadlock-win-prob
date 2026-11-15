"""Run the complete recent-replay training pipeline: python -m dwp.cli.train."""

import argparse
import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

import torch

from dwp.data.deadlock_api import GameMode, MatchMode
from dwp.data.planning import read_plan
from dwp.data.preparation import prepare_matches
from dwp.graphs.parquet import load_graph_shards
from dwp.ml.tabular_features import DEFAULT_FEATURE_SET
from dwp.ml.tabular_training import (
    DEFAULT_CONTEXT_ROWS,
    DEFAULT_NUM_ESTIMATORS,
)
from dwp.ml.training import TrainingConfig, train_model


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Parquet file build-age window (not match age)",
    )
    parser.add_argument(
        "--matches",
        type=int,
        default=20,
        help="Maximum usable matches to sample (at least 3)",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        help="Candidate query/attempt bound (default max(100, 20 * matches))",
    )
    parser.add_argument(
        "--every", type=int, default=4096, help="Ticks between graph snapshots"
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=16,
        help="Maximum retained snapshots per match",
    )
    parser.add_argument(
        "--match-modes",
        nargs="+",
        choices=get_args(MatchMode.__value__),
        default=["Ranked"],
    )
    parser.add_argument(
        "--game-modes",
        nargs="+",
        choices=get_args(GameMode.__value__),
        default=["Normal"],
    )
    parser.add_argument(
        "--data-version", help="Optional matching boon-data catalog version"
    )
    parser.add_argument(
        "--prepared-data",
        type=Path,
        help="Read Parquet graph shards instead of downloading replays",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        help="Expected preparation plan (required with --prepared-data)",
    )
    parser.add_argument("--model", choices=("graph", "kumo"), default="graph")
    parser.add_argument(
        "--model-version", default="v1.0.0", help="Model version (vMAJOR.MINOR.PATCH)"
    )
    parser.add_argument(
        "--features",
        choices=("baseline", "heroes", "heroes-items"),
        help=f"Kumo feature set (default: {DEFAULT_FEATURE_SET}). heroes-items requires --data-version.",
    )
    parser.add_argument(
        "--context-rows",
        type=int,
        default=DEFAULT_CONTEXT_ROWS,
        help="Maximum Kumo context rows",
    )
    parser.add_argument(
        "--num-estimators",
        type=int,
        default=DEFAULT_NUM_ESTIMATORS,
        help="Number of Kumo ensemble estimators",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Graphs per tensor batch for training, validation, and test evaluation",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args(argv)
    features = args.features or DEFAULT_FEATURE_SET
    try:
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            bins=args.bins,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            device=args.device,
            model={
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            },
        )
        if (
            min(args.days, args.every, args.max_snapshots, args.threads) < 1
            or args.matches < 3
        ):
            raise ValueError(
                "days/every/max-snapshots/threads must be positive and matches >= 3"
            )
        if args.model == "kumo":
            if args.prepared_data is None and args.matches < 4:
                raise ValueError("Kumo requires --matches >= 4")
            if args.context_rows < 2 or args.num_estimators < 1:
                raise ValueError(
                    "Kumo requires --context-rows >= 2 and --num-estimators >= 1"
                )
        if (
            args.model == "kumo"
            and features == "heroes-items"
            and not args.data_version
        ):
            raise ValueError("heroes-items requires --data-version")
        if args.model == "graph" and args.features not in (None, "baseline"):
            raise ValueError("--features applies only to Kumo")
        if args.candidate_limit is not None and args.candidate_limit < args.matches:
            raise ValueError("candidate-limit must be at least matches")
        if (args.prepared_data is None) != (args.plan is None):
            raise ValueError("--prepared-data and --plan must be supplied together")
        if args.output.exists():
            raise ValueError(
                f"output already exists: {args.output}; choose a new directory"
            )
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    torch.set_num_threads(args.threads)
    os.environ.setdefault("BOON_TICK_SEGMENTS", "1")
    data_config = vars(args)
    if args.prepared_data is not None:
        plan = read_plan(args.plan)
        matches = load_graph_shards(args.prepared_data, plan)
        data_config = {
            "prepared_data": str(args.prepared_data),
            "preparation_plan": plan,
        }
    else:
        matches = prepare_matches(
            days=args.days,
            matches=args.matches,
            every=args.every,
            max_snapshots=args.max_snapshots,
            seed=args.seed,
            candidate_limit=args.candidate_limit,
            match_modes=args.match_modes,
            game_modes=args.game_modes,
            data_version=args.data_version if args.model == "graph" else None,
        )
    if args.model == "kumo":
        from dwp.ml.tabular_training import train_tabular

        result = train_tabular(
            matches,
            args.output,
            model_version=f"model/kumo-win/{args.model_version}",
            device=None if args.device == "auto" else args.device,
            seed=config.seed,
            bins=config.bins,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            context_rows=args.context_rows,
            num_estimators=args.num_estimators,
            features=features,
            data_version=args.data_version,
        )
    else:
        result = train_model(matches, args.output, config, data_config=data_config)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "test_log_loss": result["test"].log_loss,
                "test_ece": result["test"].ece,
            }
        )
    )


if __name__ == "__main__":
    main()
