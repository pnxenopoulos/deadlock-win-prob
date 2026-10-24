"""Construct, select, download, and parse Valve replays."""

import bz2
from collections.abc import Sequence
from compression import zstd
from tempfile import NamedTemporaryFile

import polars as pl
import requests
from boon import Demo

from dwp.data._http import USER_AGENT


def add_replay_urls(match_salts: pl.DataFrame) -> pl.DataFrame:
    """Add replay_url, leaving it null when any required field is missing."""
    return match_salts.with_columns(
        pl.format(
            "http://replay{}.valve.net/1422450/{}_{}.dem.bz2",
            "cluster_id",
            "match_id",
            "replay_salt",
        ).alias("replay_url")
    )


def select_replays(
    replays: pl.DataFrame,
    match_ids: Sequence[int],
    sample_size: int | None = None,
    seed: int | None = None,
) -> pl.DataFrame:
    """Return match_id/replay_url pairs for the requested matches.

    Keep the first nonempty URL per match. Sample up to sample_size distinct
    matches without replacement; None returns all matches. A seed makes the
    sample reproducible.
    """
    if sample_size is not None and sample_size < 0:
        raise ValueError("sample_size must be nonnegative")
    selected = (
        replays.select("match_id", "replay_url")
        .filter(pl.col("match_id").is_in(match_ids), pl.col("replay_url") != "")
        .unique(subset="match_id", keep="first", maintain_order=True)
    )
    if sample_size is None:
        return selected
    return selected.sample(n=min(sample_size, selected.height), shuffle=True, seed=seed)


def download_replay(replay_url: str, session: requests.Session | None = None) -> bytes:
    """Download and decompress a replay, optionally reusing a requests session."""
    get = requests.get if session is None else session.get
    response = get(replay_url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    content = response.content
    return (
        zstd.decompress(content)
        if content.startswith(b"\x28\xb5\x2f\xfd")
        else bz2.decompress(content)
    )


def load_replay(
    replay_url: str,
    session: requests.Session | None = None,
    *,
    preload: bool = False,
) -> Demo:
    """Return a Boon Demo backed by an automatically cleaned temporary file.

    The demo retains its memory mapping for lazy parsing after the file closes.
    Set preload=True to load combat datasets during construction.
    """
    with NamedTemporaryFile(suffix=".dem") as replay:
        replay.write(download_replay(replay_url, session=session))
        replay.flush()
        return Demo(replay.name, preload=preload)
