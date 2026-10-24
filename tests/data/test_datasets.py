"""Whole-match partitioning and explicit label semantics."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dwp.data.datasets import split_matches, winner_label


def test_split_retains_rows_without_match_overlap() -> None:
    rows = pl.DataFrame(
        {"match_id": [i for i in range(30) for _ in range(i + 1)], "value": range(465)}
    )
    splits = split_matches(rows)
    parts = (splits.train, splits.validation, splits.test)
    ids = [set(part["match_id"]) for part in parts]
    assert [len(part) for part in ids] == [24, 3, 3]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert_frame_equal(pl.concat(parts).sort("value"), rows)
    reordered = split_matches(rows.reverse())
    assert set(reordered.train["match_id"]) == ids[0]
    assert set(split_matches(rows, seed=3).train["match_id"]) != ids[0]


def test_chronological_split_uses_match_time() -> None:
    rows = pl.DataFrame(
        {"match_id": [2, 2, 1, 3, 4, 5], "started_at": [10, 10, 20, 30, 40, 50]}
    )
    splits = split_matches(rows, order_by="started_at")
    assert set(splits.train["match_id"]) == {2, 1, 3}
    assert splits.validation["match_id"].to_list() == [4]
    assert splits.test["match_id"].to_list() == [5]
    for invalid in (
        rows.with_columns(pl.lit(None).alias("started_at")),
        rows.with_row_index().with_columns(pl.col("index").alias("started_at")),
    ):
        with pytest.raises(ValueError, match="one non-null"):
            split_matches(invalid, order_by="started_at")


@pytest.mark.parametrize(
    "train,validation",
    [(0, 0.1), (0.8, 0), (0.9, 0.1), (float("nan"), 0.1), (0.8, float("inf"))],
)
def test_bad_fractions(train: float, validation: float) -> None:
    with pytest.raises(ValueError, match="fractions"):
        split_matches(pl.DataFrame({"match_id": [1, 2, 3]}), train, validation)


@pytest.mark.parametrize("ids", [[], [1, 1, 2], [1, 2, None], [1.0, 2.0, 3.0]])
def test_bad_ids(ids: list) -> None:
    with pytest.raises(ValueError):
        split_matches(pl.DataFrame({"match_id": ids}))


def test_small_dataset_has_three_nonempty_partitions() -> None:
    parts = split_matches(pl.DataFrame({"match_id": [1, 2, 3]}))
    assert [part.height for part in (parts.train, parts.validation, parts.test)] == [
        1,
        1,
        1,
    ]


def test_winner_labels() -> None:
    assert winner_label(3) == 1
    assert winner_label(2) == 0
    for team in (0, 1, 4):
        with pytest.raises(ValueError):
            winner_label(team)
