"""Store parsed graph shards as Parquet and combine complete preparation plans."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, cast, overload

import polars as pl

from dwp.data.planning import PreparationPlan
from dwp.data.preparation import ReplayMatch
from dwp.graphs.snapshots import GraphSnapshot

logger = logging.getLogger(__name__)

_TABLES = ("global_info", "players", "objectives", "ziplines")
_KEYS = ("_match_id", "_snapshot_tick")
_MATCH_SCHEMA = {
    "match_id": pl.UInt64,
    "label": pl.UInt8,
    "replay_url": pl.String,
    "build": pl.Int64,
    "game_over_tick": pl.Int64,
    "data_version": pl.String,
    "ticks": pl.List(pl.Int64),
}


class _ShardMetadata(TypedDict):
    format_version: int
    plan_id: str
    shard_id: int
    shard_count: int
    matches: int
    snapshots: int


@dataclass(frozen=True, slots=True)
class _StoredSnapshots(Sequence[GraphSnapshot]):
    """Read only requested row ranges; shared temporary files own the raw data."""

    workspace: TemporaryDirectory[str]
    tables: dict[str, pl.LazyFrame]
    ranges: dict[str, dict[tuple[int, int], tuple[int, int]]]
    match_id: int
    build: int
    data_version: str | None
    ticks: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.ticks)

    @overload
    def __getitem__(self, index: int) -> GraphSnapshot: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[GraphSnapshot]: ...

    def __getitem__(
        self, index: int | slice
    ) -> GraphSnapshot | Sequence[GraphSnapshot]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        tick = self.ticks[index]
        return GraphSnapshot(
            tick=tick,
            match_id=self.match_id,
            build=self.build,
            data_version=self.data_version,
            **{
                name: table.slice(*self.ranges[name].get((self.match_id, tick), (0, 0)))
                .collect()
                .drop(_KEYS)
                for name, table in self.tables.items()
            },
        )


def _validate_match(match: ReplayMatch, plan: PreparationPlan) -> None:
    if match.label not in (0, 1) or not 1 <= len(match.graphs) <= plan["max_snapshots"]:
        raise ValueError("matches need binary labels and a bounded nonempty graph list")
    if len({g.tick for g in match.graphs}) != len(match.graphs):
        raise ValueError("duplicate snapshot ticks in a match")
    for graph in match.graphs:
        if (
            graph.match_id != match.match_id
            or graph.build != match.build
            or graph.data_version != plan["data_version"]
            or not 0 <= graph.tick < match.game_over_tick
        ):
            raise ValueError(
                "graph metadata differs from match/plan or includes post-game state"
            )
        if graph.global_info.height != 1 or graph.players.is_empty():
            raise ValueError(
                "each snapshot needs one global node and player observations"
            )


def save_graph_shard(
    matches: Sequence[ReplayMatch],
    output: Path,
    plan: PreparationPlan,
    shard_id: int,
) -> None:
    """Save raw node tables, labels, and metadata; never fitted/encoded features.

    Empty shards are valid when all assigned replays fail. dataset.json is
    written last as the completion marker. Existing outputs are not overwritten.
    """
    if not 0 <= shard_id < len(plan["shards"]):
        raise ValueError("invalid shard ID")
    assignment = plan["shards"][shard_id]
    allowed = {r["match_id"]: r["replay_url"] for r in assignment["replays"]}
    if len(matches) > assignment["matches"] or len(
        {m.match_id for m in matches}
    ) != len(matches):
        raise ValueError("duplicate matches or shard quota exceeded")
    for match in matches:
        if allowed.get(match.match_id) != match.replay_url:
            raise ValueError("match was not assigned to this shard")
        _validate_match(match, plan)
    output.mkdir(parents=True, exist_ok=False)
    pl.DataFrame(
        [
            {
                "match_id": m.match_id,
                "label": m.label,
                "replay_url": m.replay_url,
                "build": m.build,
                "game_over_tick": m.game_over_tick,
                "data_version": plan["data_version"],
                "ticks": [g.tick for g in m.graphs],
            }
            for m in matches
        ],
        schema=_MATCH_SCHEMA,
    ).write_parquet(output / "matches.parquet", compression="zstd")
    for name in _TABLES:
        frames = []
        for match in matches:
            for graph in match.graphs:
                table: pl.DataFrame = getattr(graph, name)
                if set(_KEYS) & set(table.columns):
                    raise ValueError("graph tables use reserved serialization columns")
                frames.append(
                    table.with_columns(
                        pl.lit(match.match_id, dtype=pl.UInt64).alias(_KEYS[0]),
                        pl.lit(graph.tick, dtype=pl.Int64).alias(_KEYS[1]),
                    )
                )
        combined = (
            pl.concat(frames, how="diagonal_relaxed")
            if frames
            else pl.DataFrame(schema={_KEYS[0]: pl.UInt64, _KEYS[1]: pl.Int64})
        )
        combined.write_parquet(output / f"{name}.parquet", compression="zstd")
    metadata: _ShardMetadata = {
        "format_version": 1,
        "plan_id": plan["plan_id"],
        "shard_id": shard_id,
        "shard_count": len(plan["shards"]),
        "matches": len(matches),
        "snapshots": sum(len(m.graphs) for m in matches),
    }
    (output / "dataset.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def load_graph_shards(root: Path, plan: PreparationPlan) -> list[ReplayMatch]:
    """Load all planned workers; reject missing, mixed, duplicate, or invalid data.

    root may be one shard directory or a parent containing per-artifact shard
    directories. Match IDs, labels, nullable nested features, and node order are
    retained. Raw tables are converted one at a time to temporary uncompressed
    IPC files, then read in snapshot-sized slices. Graphs are a lazy sequence,
    not a retained list of DataFrames. Temporary files live as long as the
    returned matches and are removed on release. Allow local disk space for
    uncompressed node tables. No encoded features or fitted statistics are saved.
    """
    descriptors = (
        [root / "dataset.json"]
        if (root / "dataset.json").is_file()
        else sorted(root.glob("*/dataset.json"))
    )
    found: dict[int, Path] = {}
    metadata_by_shard: dict[int, _ShardMetadata] = {}
    for path in descriptors:
        metadata = cast(_ShardMetadata, json.loads(path.read_text(encoding="utf-8")))
        shard = metadata["shard_id"]
        if (
            metadata["format_version"] != 1
            or metadata["plan_id"] != plan["plan_id"]
            or metadata["shard_count"] != len(plan["shards"])
        ):
            raise ValueError("mixed preparation plans or unsupported shard format")
        if shard in found or not 0 <= shard < len(plan["shards"]):
            raise ValueError("duplicate or unexpected shard ID")
        found[shard], metadata_by_shard[shard] = path.parent, metadata
    if set(found) != set(range(len(plan["shards"]))):
        raise ValueError("missing prepared shards; all planned workers must complete")
    workspace = TemporaryDirectory(prefix="dwp-graphs-")
    matches: list[ReplayMatch] = []
    seen: set[int] = set()
    logger.info("Indexing %d prepared shards into disk-backed snapshots", len(found))
    for shard, folder in sorted(found.items()):
        assignment = plan["shards"][shard]
        allowed = {r["match_id"]: r["replay_url"] for r in assignment["replays"]}
        rows = pl.read_parquet(folder / "matches.parquet").to_dicts()
        metadata = metadata_by_shard[shard]
        if len(rows) != metadata["matches"] or len(rows) > assignment["matches"]:
            raise ValueError("shard match count differs from manifest/quota")
        expected = {(row["match_id"], tick) for row in rows for tick in row["ticks"]}
        if len(expected) != metadata["snapshots"]:
            raise ValueError("shard snapshot count differs from manifest")
        tables: dict[str, pl.LazyFrame] = {}
        ranges: dict[str, dict[tuple[int, int], tuple[int, int]]] = {}
        for name in _TABLES:
            # Only one decompressed shard table is resident during conversion.
            table = pl.read_parquet(folder / f"{name}.parquet")
            runs = (
                table.select(*_KEYS, pl.struct(_KEYS).rle_id().alias("_run"))
                .group_by("_run", maintain_order=True)
                .agg(*(pl.col(key).first() for key in _KEYS), pl.len().alias("length"))
            )
            spans: dict[tuple[int, int], tuple[int, int]] = {}
            offset = 0
            for match_id, tick, length in runs.select(*_KEYS, "length").iter_rows():
                key = (match_id, tick)
                if key not in expected:
                    raise ValueError(
                        "feature table contains unlisted matches or snapshots"
                    )
                if key in spans:
                    raise ValueError(
                        "snapshot rows must be contiguous within each table"
                    )
                spans[key] = (offset, length)
                offset += length
            target = Path(workspace.name) / f"{shard}-{name}.arrow"
            table.write_ipc(target, compression="uncompressed", record_batch_size=4096)
            del table
            tables[name], ranges[name] = pl.scan_ipc(target), spans
        for row in rows:
            match_id = row["match_id"]
            if match_id in seen:
                raise ValueError("duplicate match IDs across prepared shards")
            if (
                allowed.get(match_id) != row["replay_url"]
                or row["data_version"] != plan["data_version"]
            ):
                raise ValueError("prepared match differs from its plan assignment")
            ticks = row["ticks"]
            if (
                row["label"] not in (0, 1)
                or not 1 <= len(ticks) <= plan["max_snapshots"]
                or any(not 0 <= tick < row["game_over_tick"] for tick in ticks)
            ):
                raise ValueError(
                    "matches need binary labels and bounded pre-game-over snapshots"
                )
            if len(set(ticks)) != len(ticks):
                raise ValueError("duplicate snapshot ticks in a match")
            if any(
                ranges["global_info"].get((match_id, tick), (0, 0))[1] != 1
                or ranges["players"].get((match_id, tick), (0, 0))[1] < 1
                for tick in ticks
            ):
                raise ValueError(
                    "each snapshot needs one global node and player observations"
                )
            graphs = _StoredSnapshots(
                workspace,
                tables,
                ranges,
                match_id,
                row["build"],
                row["data_version"],
                tuple(ticks),
            )
            match = ReplayMatch(
                match_id,
                row["label"],
                row["replay_url"],
                row["build"],
                row["game_over_tick"],
                graphs,
            )
            seen.add(match_id)
            matches.append(match)
        logger.info(
            "Indexed shard %d/%d: %d matches, %d snapshots (%d matches total)",
            shard + 1,
            len(found),
            len(rows),
            metadata["snapshots"],
            len(matches),
        )
    return sorted(matches, key=lambda match: match.match_id)
