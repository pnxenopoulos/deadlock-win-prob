"""Partition rows by whole match and encode Boon's native winner labels."""

from dataclasses import dataclass
from math import isfinite
from random import Random

import polars as pl


@dataclass(frozen=True, slots=True)
class MatchSplits:
    """Disjoint match partitions, retaining every input row and column."""

    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame


def split_matches(
    rows: pl.DataFrame,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    *,
    seed: int = 42,
    order_by: str | None = None,
) -> MatchSplits:
    """Split distinct match IDs, never individual snapshots (default 80/10/10).

    With order_by, put earlier matches in train and later matches in test;
    that column must have one non-null value per match (e.g. start time).
    Otherwise shuffle sorted IDs with seed, independently of input row order.
    Each partition gets at least one match; small datasets approximate fractions.
    """
    if (
        not isfinite(train_fraction)
        or not isfinite(validation_fraction)
        or train_fraction <= 0
        or validation_fraction <= 0
        or train_fraction + validation_fraction >= 1
    ):
        raise ValueError("train/validation fractions must be positive and sum to < 1")
    ids = rows["match_id"]
    if ids.null_count() or not ids.dtype.is_integer():
        raise ValueError("match_id must contain non-null integers")
    matches = rows.select("match_id").unique().sort("match_id")
    if matches.height < 3:
        raise ValueError("at least three distinct matches are required")
    if order_by is not None:
        times = rows.group_by("match_id").agg(
            pl.col(order_by).n_unique().alias("n"),
            pl.col(order_by).first().alias("time"),
        )
        if times.filter((pl.col("n") != 1) | pl.col("time").is_null()).height:
            raise ValueError("order_by must have one non-null value per match")
        matches = times.sort("time", "match_id").select("match_id")
    match_ids: list[int] = matches["match_id"].to_list()
    if order_by is None:
        Random(seed).shuffle(match_ids)
    n_validation = max(1, round(len(match_ids) * validation_fraction))
    n_test = max(1, round(len(match_ids) * (1 - train_fraction - validation_fraction)))
    n_train = len(match_ids) - n_validation - n_test
    if n_train < 1:
        raise ValueError("too few matches for these split fractions")
    partitions = (
        match_ids[:n_train],
        match_ids[n_train : n_train + n_validation],
        match_ids[n_train + n_validation :],
    )
    return MatchSplits(*(rows.filter(pl.col("match_id").is_in(p)) for p in partitions))


def winner_label(winning_team_num: int) -> int:
    """Return 1 for Archmother (3), 0 for Hidden King (2); reject other outcomes.

    These are Boon native team numbers, not deadlock-api's Team0/Team1 enum.
    """
    if winning_team_num not in (2, 3):
        raise ValueError("winner must be Hidden King (2) or Archmother (3)")
    return int(winning_team_num == 3)
