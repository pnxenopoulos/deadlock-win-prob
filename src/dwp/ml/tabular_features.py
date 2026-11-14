"""Define and reconstruct the lineup and inventory columns in a Kumo context."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import polars as pl
from boon import data

from dwp.graphs.snapshots import GraphSnapshot

type FeatureSet = Literal["baseline", "heroes", "heroes-items"]
DEFAULT_FEATURE_SET: FeatureSet = "heroes-items"

_SLOTS = {
    "EItemSlotType_WeaponMod": "weapon",
    "EItemSlotType_Armor": "vitality",
    "EItemSlotType_Tech": "spirit",
}


@dataclass(frozen=True, slots=True)
class ItemCatalog:
    """Classify recorded upgrade IDs with one verified boon-data release."""

    version: str
    checksum: str
    item_groups: dict[int, str]
    non_items: frozenset[int]

    @classmethod
    def load(cls, version: str) -> ItemCatalog:
        """Read item categories and tiers from the verified abilities catalog."""
        content = data.catalog_path("abilities", version).read_bytes()
        groups = {}
        non_items = set()
        for record in json.loads(content)["records"]:
            ability_id = record["ability_id"]
            definition = record["definition"]
            ability_type = definition.get("m_eAbilityType")
            if ability_type == "EAbilityType_Item":
                slot = _SLOTS.get(definition.get("m_eItemSlotType"))
                raw_tier = definition.get("m_iItemTier")
                tier = (
                    raw_tier.removeprefix("EModTier_")
                    if isinstance(raw_tier, str)
                    else ""
                )
                if slot and tier.isdigit():
                    groups[ability_id] = f"{slot}_tier_{tier}"
            elif ability_type is not None:
                non_items.add(ability_id)
        if not groups:
            raise ValueError(f"boon-data {version} has no categorized items")
        return cls(version, sha256(content).hexdigest(), groups, frozenset(non_items))

    def counts(self, players: pl.DataFrame) -> dict[str, int]:
        """Count current items per group. Omit counts for incomplete inventories."""
        if players.is_empty():
            return {}
        groups: Counter[str] = Counter()
        for inventory in players["upgrade_ids"]:
            if inventory is None:
                return {}
            for token in set(inventory) - self.non_items:
                group = self.item_groups.get(token)
                if group is None:
                    return {}
                groups[group] += 1
        return dict.fromkeys(self.item_groups.values(), 0) | dict(groups)


@dataclass(frozen=True, slots=True)
class LineupFeatures:
    """Training vocabulary and optional item catalog used for both fit and predict."""

    hero_ids: tuple[int, ...] = ()
    items: ItemCatalog | None = None

    @classmethod
    def fit(
        cls, graphs: Iterable[GraphSnapshot], items: ItemCatalog | None = None
    ) -> LineupFeatures:
        """Collect hero IDs from observed training lineups."""
        ids = {
            hero
            for graph in graphs
            for hero in graph.players.filter(pl.col("team_num").is_in([2, 3]))[
                "hero_id"
            ]
            if hero is not None and hero > 0
        }
        if not ids:
            raise ValueError("training snapshots have no observed hero IDs")
        return cls(tuple(sorted(ids)), items)

    @property
    def fields(self) -> tuple[str, ...]:
        """Return the stable feature suffixes for one team."""
        return (
            *(f"hero_{hero}" for hero in self.hero_ids),
            *(
                (
                    f"item_{group}_count"
                    for group in sorted(set(self.items.item_groups.values()))
                )
                if self.items is not None
                else ()
            ),
        )

    def values(self, players: pl.DataFrame) -> dict[str, int | None]:
        """Encode one team's lineup and current item counts."""
        ids = set(players["hero_id"]) if self.hero_ids else set()
        known = bool(ids) and all(hero is not None and hero > 0 for hero in ids)
        stats = {
            f"hero_{hero}": 1 if hero in ids else 0 if known else None
            for hero in self.hero_ids
        }
        if self.items is not None:
            stats.update(
                {
                    f"item_{group}_count": count
                    for group, count in self.items.counts(players).items()
                }
            )
        return stats
