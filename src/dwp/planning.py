"""Create and validate deterministic, disjoint replay-preparation plans."""

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

from dwp.deadlock_api import GameMode, MatchMode
from dwp.preparation import select_candidates


class ReplayCandidate(TypedDict):
    match_id: int
    replay_url: str


class ShardPlan(TypedDict):
    shard_id: int
    matches: int
    replays: list[ReplayCandidate]


class PreparationPlan(TypedDict):
    format_version: int
    plan_id: str
    requested_matches: int
    every: int
    max_snapshots: int
    seed: int
    data_version: str | None
    source: dict[str, object]
    shards: list[ShardPlan]


def _fingerprint(plan: PreparationPlan) -> str:
    payload = {name: value for name, value in plan.items() if name != "plan_id"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def plan_matches(
    *,
    days: int = 1,
    matches: int = 20,
    workers: int = 4,
    seed: int = 42,
    every: int = 4096,
    max_snapshots: int = 16,
    candidate_limit: int | None = None,
    match_modes: Sequence[MatchMode] = ("Ranked",),
    game_modes: Sequence[GameMode] = ("Normal",),
    data_version: str | None = None,
) -> PreparationPlan:
    """Discover once, shuffle once, then assign disjoint pools and fair quotas.

    Activate at most min(workers, matches, eligible URLs) jobs. Spare URLs are
    assigned to the same workers for replacement attempts; no worker discovers
    or samples another worker's matches. Limit workers to Actions' matrix size.
    """
    if not 1 <= workers <= 256 or min(every, max_snapshots) < 1:
        raise ValueError(
            "workers must be in [1, 256]; every/max_snapshots must be positive"
        )
    if data_version is not None and not data_version:
        raise ValueError("data_version must be nonempty")
    replays = select_candidates(
        days=days,
        matches=matches,
        seed=seed,
        candidate_limit=candidate_limit,
        match_modes=match_modes,
        game_modes=game_modes,
    )
    if replays.height < 3:
        raise ValueError("at least three candidates with replay URLs are required")
    target = min(matches, replays.height)
    active = min(workers, target)
    shards: list[ShardPlan] = [
        {
            "shard_id": i,
            "matches": target // active + int(i < target % active),
            "replays": [],
        }
        for i in range(active)
    ]
    for i, (match_id, url) in enumerate(replays.iter_rows()):
        shards[i % active]["replays"].append({"match_id": match_id, "replay_url": url})
    plan: PreparationPlan = {
        "format_version": 1,
        "plan_id": "",
        "requested_matches": matches,
        "every": every,
        "max_snapshots": max_snapshots,
        "seed": seed,
        "data_version": data_version,
        "source": {
            "days": days,
            "candidate_limit": candidate_limit,
            "match_modes": list(match_modes),
            "game_modes": list(game_modes),
            "workers": workers,
        },
        "shards": shards,
    }
    plan["plan_id"] = _fingerprint(plan)
    logging.getLogger(__name__).info(
        "Planned %d candidate URLs across %d workers for up to %d matches",
        replays.height,
        active,
        target,
    )
    return plan


def write_plan(plan: PreparationPlan, path: Path) -> None:
    """Write a new plan, refusing to overwrite an existing assignment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        file.write(json.dumps(plan, indent=2) + "\n")


def read_plan(path: Path) -> PreparationPlan:
    """Validate a plan's identity, quotas, and nonoverlapping candidate pools."""
    plan = cast(PreparationPlan, json.loads(path.read_text(encoding="utf-8")))
    if plan["format_version"] != 1 or plan["plan_id"] != _fingerprint(plan):
        raise ValueError("plan format or fingerprint is invalid")
    shards = plan["shards"]
    if not 1 <= len(shards) <= 256 or [s["shard_id"] for s in shards] != list(
        range(len(shards))
    ):
        raise ValueError("plan shard IDs must be consecutive from zero")
    if min(plan["every"], plan["max_snapshots"]) < 1:
        raise ValueError("plan sampling settings must be positive")
    ids: set[int] = set()
    for shard in shards:
        if not 1 <= shard["matches"] <= len(shard["replays"]):
            raise ValueError("each shard needs candidates sufficient for its quota")
        for candidate in shard["replays"]:
            match_id = candidate["match_id"]
            if match_id in ids or match_id < 0 or not candidate["replay_url"]:
                raise ValueError(
                    "plan contains duplicate IDs or invalid replay candidates"
                )
            ids.add(match_id)
    if not 3 <= sum(s["matches"] for s in shards) <= plan["requested_matches"]:
        raise ValueError("plan must target between three and requested_matches matches")
    return plan
