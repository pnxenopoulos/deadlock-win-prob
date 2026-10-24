"""Disjoint worker plans, deterministic quotas, and portable plan identity."""

import json
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest

from dwp.data.planning import plan_matches, read_plan, write_plan


@pytest.fixture
def candidates(monkeypatch: pytest.MonkeyPatch) -> Mock:
    source = Mock(
        return_value=pl.DataFrame(
            {
                "match_id": list(range(100, 112)),
                "replay_url": [
                    f"https://example.com/{i}.dem.bz2" for i in range(100, 112)
                ],
            }
        )
    )
    monkeypatch.setattr("dwp.data.planning.select_candidates", source)
    return source


def test_planning_is_disjoint_balanced_and_deterministic(
    candidates: Mock, tmp_path: Path
) -> None:
    plan = plan_matches(matches=7, workers=3, every=1234, max_snapshots=2)
    assert [s["matches"] for s in plan["shards"]] == [3, 2, 2]
    pools = [{r["match_id"] for r in s["replays"]} for s in plan["shards"]]
    assert (
        not pools[0] & pools[1] and not pools[1] & pools[2] and not pools[0] & pools[2]
    )
    assert set.union(*pools) == set(range(100, 112))
    assert plan_matches(matches=7, workers=3, every=1234, max_snapshots=2) == plan
    assert plan_matches(matches=7, workers=3, every=4321)["plan_id"] != plan["plan_id"]
    path = tmp_path / "plan.json"
    write_plan(plan, path)
    assert read_plan(path) == plan
    with pytest.raises(FileExistsError):
        write_plan(plan, path)
    changed = json.loads(path.read_text())
    changed["seed"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fingerprint"):
        read_plan(path)


def test_workers_are_capped_by_target_and_available_urls(candidates: Mock) -> None:
    assert len(plan_matches(matches=3, workers=100)["shards"]) == 3
    candidates.return_value = candidates.return_value.head(5)
    plan = plan_matches(matches=20, workers=10)
    assert len(plan["shards"]) == 5
    assert sum(s["matches"] for s in plan["shards"]) == 5
    candidates.return_value = candidates.return_value.head(2)
    with pytest.raises(ValueError, match="three candidates"):
        plan_matches(matches=3)


@pytest.mark.parametrize(
    "settings",
    [
        {"workers": 0},
        {"workers": 257},
        {"every": 0},
        {"max_snapshots": 0},
        {"data_version": ""},
    ],
)
def test_invalid_plan_options_do_not_query(candidates: Mock, settings: dict) -> None:
    with pytest.raises(ValueError):
        plan_matches(**settings)
    candidates.assert_not_called()
