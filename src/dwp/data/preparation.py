"""Collect replay-backed, pre-victory graph snapshots for training."""

import logging
from collections.abc import Sequence
from compression.zstd import ZstdError
from dataclasses import dataclass
from random import Random

import polars as pl
import requests

from dwp.data.datasets import winner_label
from dwp.data.deadlock_api import (
    GameMode,
    MatchMode,
    load_latest_match_salts,
    load_recent_matches,
)
from dwp.data.replays import add_replay_urls, load_replay, select_replays
from dwp.graphs.snapshots import GraphSnapshot, iter_graphs

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReplayMatch:
    """Graphs and outcome metadata kept separate to avoid target leakage."""

    match_id: int
    label: int
    replay_url: str
    build: int
    game_over_tick: int
    graphs: list[GraphSnapshot]


def select_candidates(
    *,
    days: int = 1,
    matches: int = 20,
    seed: int = 42,
    candidate_limit: int | None = None,
    match_modes: Sequence[MatchMode] = ("Ranked",),
    game_modes: Sequence[GameMode] = ("Normal",),
) -> pl.DataFrame:
    """Query recent matches/salts once and return a seeded candidate URL order."""
    if days < 1 or matches < 3:
        raise ValueError("days must be positive and matches >= 3")
    limit = candidate_limit if candidate_limit is not None else max(100, matches * 20)
    if limit < matches:
        raise ValueError("candidate_limit must be at least matches")
    logger.info(
        "Loading up to %d recent candidate matches (file age <= %d days)", limit, days
    )
    recent = load_recent_matches(days, match_modes, game_modes, limit=limit)
    ids: list[int] = (
        recent.filter(pl.col("winning_team").is_in(["Team0", "Team1"]))["match_id"]
        .unique()
        .to_list()
    )
    if len(ids) < 3:
        raise ValueError(
            "fewer than three eligible recent matches; increase --days or --candidate-limit"
        )
    logger.info("Loading salts and selecting replay URLs for %d candidates", len(ids))
    salts = load_latest_match_salts().filter(pl.col("match_id").is_in(ids))
    return select_replays(
        add_replay_urls(salts).sort("match_id"), ids, seed=seed, sample_size=limit
    )


def prepare_replays(
    replays: pl.DataFrame,
    *,
    matches: int,
    every: int = 4096,
    max_snapshots: int = 16,
    seed: int = 42,
    data_version: str | None = None,
) -> list[ReplayMatch]:
    """Parse assigned URLs, replacing failed candidates up to a match quota.

    Workers may return zero or more matches. The final training dataset must
    contain at least three. No source discovery or feature fitting happens here.
    """
    if min(matches, every, max_snapshots) < 1:
        raise ValueError("matches/every/max_snapshots must be positive")
    if data_version is not None and not data_version:
        raise ValueError("data_version must be nonempty")
    prepared: list[ReplayMatch] = []
    with requests.Session() as session:
        for match_id, url in replays.iter_rows():
            if len(prepared) == matches:
                break
            demo = None
            try:
                logger.info(
                    "Parsing replay %s (%d/%d usable)", match_id, len(prepared), matches
                )
                demo = load_replay(url, session=session)
                if demo.match_id != match_id:
                    raise ValueError("downloaded replay has a different match ID")
                end = demo.game_over_tick
                winner = demo.winning_team_num
                if end is None or end <= 0 or winner is None:
                    raise ValueError("missing game-over tick or winner")
                label = winner_label(winner)
                graphs = list(
                    iter_graphs(
                        demo, every=every, end_tick=end - 1, data_version=data_version
                    )
                )
                # Reject recordings with no players; their global state cannot train a useful model.
                graphs = [g for g in graphs if g.tick < end and g.players.height > 0]
                if not graphs:
                    raise ValueError("no usable pre-victory snapshots")
                if len(graphs) > max_snapshots:
                    graphs = sorted(
                        Random(seed + match_id).sample(graphs, max_snapshots),
                        key=lambda g: g.tick,
                    )
                prepared.append(
                    ReplayMatch(match_id, label, url, demo.build, end, graphs)
                )

            except (
                OSError,
                ZstdError,
                EOFError,
                RuntimeError,
                ValueError,
                pl.exceptions.PolarsError,
            ) as error:
                logger.warning("Skipping replay %s: %s", match_id, error)
            finally:
                del demo
    if len(prepared) < matches:
        logger.warning(
            "Requested %d matches; using %d usable replays", matches, len(prepared)
        )
    return prepared


def prepare_matches(
    *,
    days: int = 1,
    matches: int = 20,
    every: int = 4096,
    max_snapshots: int = 16,
    seed: int = 42,
    candidate_limit: int | None = None,
    match_modes: Sequence[MatchMode] = ("Ranked",),
    game_modes: Sequence[GameMode] = ("Normal",),
    data_version: str | None = None,
) -> list[ReplayMatch]:
    """Discover and parse a bounded sample for the local end-to-end command.

    The candidate pool defaults to max(100, 20 * matches), visiting newer
    Parquet files first. This is not uniform sampling of the whole data lake.
    Graphs are sampled before game over, with at most max_snapshots per match.
    """
    if days < 1 or matches < 3 or every < 1 or max_snapshots < 1:
        raise ValueError("days/every/max_snapshots must be positive and matches >= 3")
    replays = select_candidates(
        days=days,
        matches=matches,
        seed=seed,
        candidate_limit=candidate_limit,
        match_modes=match_modes,
        game_modes=game_modes,
    )
    prepared = prepare_replays(
        replays,
        matches=matches,
        every=every,
        max_snapshots=max_snapshots,
        seed=seed,
        data_version=data_version,
    )
    if len(prepared) < 3:
        raise ValueError(
            f"only {len(prepared)} usable replays; at least three are required"
        )
    return prepared
