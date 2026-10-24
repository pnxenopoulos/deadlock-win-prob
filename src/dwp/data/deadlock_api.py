"""Load match data from deadlock-api (https://data.deadlock-api.com)."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal
from urllib.parse import urljoin

import polars as pl
import requests

from dwp.data._http import USER_AGENT

MANIFEST_URL = "https://data.deadlock-api.com/v1/manifest.json"

type MatchMode = Literal[
    "Invalid",
    "Unranked",
    "PrivateLobby",
    "CoopBot",
    "Ranked",
    "ServerTest",
    "Tutorial",
    "HeroLabs",
    "NewPlayerPlacement",
]
type GameMode = Literal[
    "Invalid",
    "Normal",
    "OneVsOneTest",
    "Sandbox",
    "StreetBrawl",
    "ExploreNYC",
    "Internal",
]


def load_latest_match_salts() -> pl.DataFrame:
    """Fetch the manifest and read its match_salts Parquet files."""
    response = requests.get(
        MANIFEST_URL, headers={"User-Agent": USER_AGENT}, timeout=30
    )
    response.raise_for_status()
    manifest = response.json()
    return pl.read_parquet(
        [
            urljoin(f"{manifest['public_url']}/", file["key"])
            for file in manifest["tables"]["match_salts"]["files"]
        ]
    )


def load_recent_matches(
    days: int = 7,
    match_modes: Sequence[MatchMode] = ("Ranked",),
    game_modes: Sequence[GameMode] = ("Normal",),
    *,
    limit: int | None = None,
) -> pl.DataFrame:
    """Return unique match_id/winning_team pairs from files built in the last days.

    Modes use their enum names and default to Ranked and Normal;
    only TeamWin outcomes are included. File age uses UTC build timestamps,
    so matches in those files may have older start times. Optional limit bounds
    the number of distinct matches collected, visiting newer files first.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    cutoff = datetime.now(UTC) - timedelta(days=days)
    response = requests.get(
        MANIFEST_URL, headers={"User-Agent": USER_AGENT}, timeout=30
    )
    response.raise_for_status()
    manifest = response.json()
    urls = [
        urljoin(f"{manifest['public_url']}/", file["key"])
        for file in sorted(
            manifest["tables"]["match_player"]["files"],
            key=lambda f: f["built_at"],
            reverse=True,
        )
        if datetime.fromisoformat(file["built_at"]) >= cutoff
    ]
    if not urls:
        return pl.DataFrame(schema={"match_id": pl.UInt64, "winning_team": pl.String})
    matches = pl.scan_parquet(urls).filter(
        pl.col("match_mode").is_in(match_modes),
        pl.col("match_outcome") == "TeamWin",
    )
    if game_modes is not None:
        matches = matches.filter(pl.col("game_mode").is_in(game_modes))
    selected = matches.select("match_id", "winning_team").unique(maintain_order=True)
    if limit is not None:
        selected = selected.head(limit)
    return selected.collect(engine="streaming")
