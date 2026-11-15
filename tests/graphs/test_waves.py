"""Use only the recorded live wave at each sampled tick."""

import polars as pl

from dwp.graphs.waves import wave_features


def test_front_uses_enemy_base_and_only_selected_team_lane_and_tick() -> None:
    ticks = pl.DataFrame({"tick": [9, 10]})
    troops = pl.DataFrame(
        {
            "tick": [10] * 5,
            "team_num": [2, 2, 3, 2, 2],
            "lane": [1, 1, 1, 4, 1],
            "trooper_type": ["trooper"] * 4 + ["trooper_boss"],
            "health": [121, 99, 60, 80, 900],
            "x": [0.0, 6.0, 0.0, 10.0, 10.0],
            "y": [0.0, 3.0, 0.0, 0.0, 0.0],
        }
    )
    bases = pl.DataFrame(
        {
            "tick": [10, 10],
            "observed_tick": [2, 2],
            "team_num": [2, 3],
            "objective_type": ["patron", "patron"],
            "x": [-10.0, 10.0],
            "y": [0.0, 0.0],
        }
    )

    result = wave_features(ticks, troops, bases).sort("tick")

    assert result.select(
        "wave_2_1_count", "wave_2_1_health", "wave_2_1_front_distance"
    ).rows() == [
        (0, 0, None),
        (2, 220, 5.0),
    ]
    assert result["wave_3_1_front_distance"].to_list() == [None, 10.0]
    assert result["wave_2_4_count"].to_list() == [0, 1]
