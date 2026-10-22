"""Small recorded-state graph shared by neural model tests."""

import polars as pl
import pytest

from dwp.representations import GraphSnapshot


@pytest.fixture
def graph_snapshot() -> GraphSnapshot:
    def nodes(kind: str, values: dict) -> pl.DataFrame:
        frame = pl.DataFrame(values)
        return frame.with_columns(
            pl.Series("node_id", [f"{kind}:{i}" for i in range(frame.height)]),
            pl.lit(kind).alias("node_type"),
            *[
                pl.col(c).cast(pl.Float64)
                if c in frame.columns
                else pl.lit(None, dtype=pl.Float64).alias(c)
                for c in ("x", "y", "z")
            ],
            pl.col("team_num").cast(pl.Int64)
            if "team_num" in frame.columns
            else pl.lit(None, dtype=pl.Int64).alias("team_num"),
        )

    return GraphSnapshot(
        tick=100,
        match_id=123,
        build=999,
        data_version=None,
        global_info=nodes(
            "global",
            {
                "elapsed_seconds": [10.0],
                "match_time_seconds": [None],
                "mid_boss_up": [False],
            },
        ),
        players=nodes(
            "player",
            {
                "steam_id": [101, 202],
                "team_num": [2, 3],
                "hero_id": [11, 22],
                "x": [0, 3000],
                "y": [0, 4000],
                "z": [0, 0],
                "health": [100, 50],
                "max_health": [100, 100],
                "souls": [500, 700],
                "is_alive": [True, True],
                "level": [2, 3],
                "upgrade_ids": [[1001, 1002], []],
                "item_ids": [None, []],
            },
        ).with_columns(pl.col("item_ids").cast(pl.List(pl.UInt32))),
        objectives=nodes(
            "objective",
            {
                "team_num": [2, 3],
                "objective_type": ["walker", "patron"],
                "phase": [0, 1],
                "lane": [1, 0],
                "health": [1000, 2000],
                "max_health": [2000, 2000],
                "destroyed": [False, False],
                "x": [0, 3000],
                "y": [1000, 1000],
                "z": [0, 0],
            },
        ),
        ziplines=nodes(
            "zipline",
            {
                "node_handle": [10, 20],
                "team_num": [0, None],
                "lane": [1, 1],
                "x": [1000, 2000],
                "y": [1000, 2000],
                "z": [100, 100],
                "connections": [[20], None],
                "capturable": [True, False],
            },
        ),
    )
