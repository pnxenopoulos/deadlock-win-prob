"""Tests for replay URL construction, selection, and downloading."""

import bz2
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import Mock, patch

import polars as pl
import pytest
import requests
from boon import Demo, InvalidDemoError
from polars.testing import assert_frame_equal

from dwp.data._http import USER_AGENT
from dwp.data.replays import (
    add_replay_urls,
    download_replay,
    load_replay,
    select_replays,
)


def test_add_replay_urls() -> None:
    salts = pl.DataFrame(
        {
            "match_id": [2**63 + 1, 2, 3, 4],
            "cluster_id": [123, None, 125, 126],
            "replay_salt": [2**32 - 1, 20, None, 0],
            "metadata_salt": [10, 20, 30, 40],
        },
        schema={
            "match_id": pl.UInt64,
            "cluster_id": pl.UInt32,
            "replay_salt": pl.UInt32,
            "metadata_salt": pl.UInt32,
        },
    )

    result = add_replay_urls(salts)

    assert result["replay_url"].to_list() == [
        "http://replay123.valve.net/1422450/9223372036854775809_4294967295.dem.bz2",
        None,
        None,
        "http://replay126.valve.net/1422450/4_0.dem.bz2",
    ]
    assert_frame_equal(result.drop("replay_url"), salts)
    assert "replay_url" not in salts.columns


def test_add_replay_urls_empty() -> None:
    salts = pl.DataFrame(
        schema={
            "match_id": pl.UInt64,
            "cluster_id": pl.UInt32,
            "replay_salt": pl.UInt32,
        }
    )

    result = add_replay_urls(salts)

    assert result.is_empty()
    assert result.schema["replay_url"] == pl.String


@pytest.mark.parametrize("use_session", [False, True])
def test_download_replay(use_session: bool) -> None:
    replay = b"PBDEMS2\x00\x00\xff\x01replay data"
    replay_url = "http://replay123.valve.net/1422450/1_2.dem.bz2"
    response = Mock(content=bz2.compress(replay))

    with (
        requests.Session() as session,
        patch.object(session, "get", return_value=response) as session_get,
        patch("dwp.data.replays.requests.get", return_value=response) as get,
    ):
        session.headers.update({"User-Agent": "caller-agent", "X-Custom": "keep"})
        original_headers = session.headers.copy()
        result = download_replay(replay_url, session=session if use_session else None)
        assert session.headers == original_headers

    assert result == replay
    used, unused = (session_get, get) if use_session else (get, session_get)
    used.assert_called_once_with(
        replay_url, headers={"User-Agent": USER_AGENT}, timeout=30
    )
    unused.assert_not_called()
    response.raise_for_status.assert_called_once_with()


def test_download_replay_http_error() -> None:
    response = Mock(content=b"Not found")
    response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

    with (
        patch("dwp.data.replays.requests.get", return_value=response),
        pytest.raises(requests.HTTPError, match="404"),
    ):
        download_replay("http://replay123.valve.net/1422450/1_2.dem.bz2")


def test_download_replay_invalid_compression() -> None:
    response = Mock(content=b"not a bzip2 stream")

    with (
        patch("dwp.data.replays.requests.get", return_value=response),
        pytest.raises(OSError),
    ):
        download_replay("http://replay123.valve.net/1422450/1_2.dem.bz2")


@pytest.fixture
def replays() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "match_id": [1, 1, 1, 2, 3, 4, 5, 6, None],
            "replay_url": [
                None,
                "http://replay11.valve.net/1422450/1_100.dem.bz2",
                "http://replay11.valve.net/1422450/1_101.dem.bz2",
                "http://replay12.valve.net/1422450/2_200.dem.bz2",
                None,
                "",
                "http://replay15.valve.net/1422450/5_500.dem.bz2",
                "http://replay16.valve.net/1422450/6_600.dem.bz2",
                "http://replay17.valve.net/1422450/7_700.dem.bz2",
            ],
            "extra": [0] * 9,
        },
        schema={"match_id": pl.UInt64, "replay_url": pl.String, "extra": pl.Int64},
    )


def test_select_replays(replays: pl.DataFrame) -> None:
    original = replays.clone()

    selected = select_replays(replays, [1, 1, 2, 3, 4, 5, 7])

    expected = pl.DataFrame(
        {
            "match_id": [1, 2, 5],
            "replay_url": [
                "http://replay11.valve.net/1422450/1_100.dem.bz2",
                "http://replay12.valve.net/1422450/2_200.dem.bz2",
                "http://replay15.valve.net/1422450/5_500.dem.bz2",
            ],
        },
        schema={"match_id": pl.UInt64, "replay_url": pl.String},
    )
    assert_frame_equal(selected, expected)
    assert_frame_equal(replays, original)


def test_select_replays_sample(replays: pl.DataFrame) -> None:
    match_ids = [1, 2, 3, 4, 5]

    selected = select_replays(replays, match_ids, sample_size=2, seed=42)

    assert selected.height == selected["match_id"].n_unique() == 2
    assert set(selected["match_id"].to_list()) <= {1, 2, 5}
    assert selected.columns == ["match_id", "replay_url"]
    assert_frame_equal(
        selected, select_replays(replays, match_ids, sample_size=2, seed=42)
    )


@pytest.mark.parametrize(("sample_size", "expected_count"), [(0, 0), (10, 3)])
def test_select_replays_sample_size(
    replays: pl.DataFrame, sample_size: int, expected_count: int
) -> None:
    selected = select_replays(replays, [1, 2, 3, 4, 5], sample_size=sample_size)

    assert selected.height == selected["match_id"].n_unique() == expected_count
    assert selected.schema == {"match_id": pl.UInt64, "replay_url": pl.String}


@pytest.mark.parametrize("match_ids", [[], [7], [3, 4]])
def test_select_replays_no_matches(replays: pl.DataFrame, match_ids: list[int]) -> None:
    selected = select_replays(replays, match_ids, sample_size=5)

    assert selected.is_empty()
    assert selected.schema == {"match_id": pl.UInt64, "replay_url": pl.String}


@pytest.mark.parametrize("sample_size", [None, 5])
def test_select_replays_empty(replays: pl.DataFrame, sample_size: int | None) -> None:
    selected = select_replays(replays.clear(), [1, 2], sample_size=sample_size)

    assert selected.is_empty()
    assert selected.schema == {"match_id": pl.UInt64, "replay_url": pl.String}


def test_select_replays_invalid_sample_size(replays: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="sample_size must be nonnegative"):
        select_replays(replays, [1], sample_size=-1)


@pytest.mark.parametrize("requested_preload", [False, True])
def test_load_replay(tmp_path: Path, requested_preload: bool) -> None:
    replay_bytes = b"decompressed replay bytes"
    replay_url = "http://replay123.valve.net/1422450/1_2.dem.bz2"
    expected = Mock(spec=Demo)
    paths: list[Path] = []

    def parse(path: str, *, preload: bool) -> Mock:
        paths.append(Path(path))
        assert paths[-1].suffix == ".dem"
        assert paths[-1].read_bytes() == replay_bytes
        assert preload is requested_preload
        return expected

    with (
        requests.Session() as session,
        patch(
            "dwp.data.replays.download_replay", return_value=replay_bytes
        ) as download,
        patch("dwp.data.replays.Demo", side_effect=parse),
        patch(
            "dwp.data.replays.NamedTemporaryFile",
            partial(NamedTemporaryFile, dir=tmp_path),
        ),
    ):
        demo = load_replay(replay_url, session=session, preload=requested_preload)

    assert demo is expected
    download.assert_called_once_with(replay_url, session=session)
    assert len(paths) == 1
    assert not paths[0].exists()


@pytest.mark.parametrize("http_error", [False, True])
def test_load_replay_cleans_up_on_error(tmp_path: Path, http_error: bool) -> None:
    error = requests.HTTPError("404 Not Found") if http_error else None

    with (
        patch(
            "dwp.data.replays.download_replay",
            return_value=b"invalid demo",
            side_effect=error,
        ),
        patch(
            "dwp.data.replays.NamedTemporaryFile",
            partial(NamedTemporaryFile, dir=tmp_path),
        ),
        pytest.raises(requests.HTTPError if http_error else InvalidDemoError),
    ):
        load_replay("http://replay123.valve.net/1422450/1_2.dem.bz2")

    assert not list(tmp_path.iterdir())


def test_download_zstandard_replay() -> None:
    from compression import zstd
    from unittest.mock import Mock, patch

    payload = b"PBDEMS2\0demo bytes" * 100
    response = Mock(content=zstd.compress(payload))
    with patch("dwp.data.replays.requests.get", return_value=response):
        assert download_replay("https://example.com/replay.dem.bz2") == payload
    response.raise_for_status.assert_called_once()
