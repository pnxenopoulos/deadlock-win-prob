"""Plan disjoint replay work and prepare individual Parquet graph shards."""

import argparse
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

import polars as pl

from dwp.data.deadlock_api import GameMode, MatchMode
from dwp.data.planning import plan_matches, read_plan, write_plan
from dwp.data.preparation import prepare_replays
from dwp.graphs.parquet import save_graph_shard


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser(
        "plan", help="Discover candidates once and assign workers"
    )
    planner.add_argument("--days", type=int, default=1)
    planner.add_argument("--matches", type=int, default=20)
    planner.add_argument("--workers", type=int, default=4)
    planner.add_argument("--max-parallel", type=int, default=4)
    planner.add_argument("--every", type=int, default=4096)
    planner.add_argument("--max-snapshots", type=int, default=16)
    planner.add_argument("--seed", type=int, default=42)
    planner.add_argument("--candidate-limit", type=int)
    planner.add_argument(
        "--match-modes",
        nargs="+",
        choices=get_args(MatchMode.__value__),
        default=["Ranked"],
    )
    planner.add_argument(
        "--game-modes",
        nargs="+",
        choices=get_args(GameMode.__value__),
        default=["Normal"],
    )
    planner.add_argument("--data-version")
    planner.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("shard", help="Parse one assigned candidate pool")
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--shard-id", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    os.environ.setdefault("BOON_TICK_SEGMENTS", "1")
    if args.command == "plan":
        if args.max_parallel < 1:
            parser.error("max-parallel must be positive")
        plan = plan_matches(
            days=args.days,
            matches=args.matches,
            workers=args.workers,
            every=args.every,
            max_snapshots=args.max_snapshots,
            seed=args.seed,
            candidate_limit=args.candidate_limit,
            match_modes=args.match_modes,
            game_modes=args.game_modes,
            data_version=args.data_version,
        )
        write_plan(plan, args.output)
        print(
            json.dumps(
                {
                    "matrix": {"shard": [s["shard_id"] for s in plan["shards"]]},
                    "plan_id": plan["plan_id"],
                    "max_parallel": min(args.max_parallel, len(plan["shards"])),
                }
            )
        )
    else:
        plan = read_plan(args.plan)
        if not 0 <= args.shard_id < len(plan["shards"]):
            parser.error("shard-id is outside the plan")
        shard = plan["shards"][args.shard_id]
        replays = pl.DataFrame(
            shard["replays"], schema={"match_id": pl.UInt64, "replay_url": pl.String}
        )
        matches = prepare_replays(
            replays,
            matches=shard["matches"],
            every=plan["every"],
            max_snapshots=plan["max_snapshots"],
            seed=plan["seed"],
            data_version=plan["data_version"],
        )
        save_graph_shard(matches, args.output, plan, args.shard_id)
        print(
            json.dumps(
                {
                    "shard_id": args.shard_id,
                    "matches": len(matches),
                    "target": shard["matches"],
                    "snapshots": sum(len(m.graphs) for m in matches),
                }
            )
        )


if __name__ == "__main__":
    main()
