import catalog from "./data/catalog.json";
import initialPresets from "./data/presets.json";
import type { RawGraph, RawNode, RawValue } from "./ml/types";

export const TEAMS = [3, 2] as const;
export const TEAM_NAMES: Record<number, string> = {
  3: "Archmother",
  2: "Hidden King",
  0: "Neutral",
};
export const TEAM_COLORS: Record<number, string> = {
  3: "#e08438",
  2: "#5b8df7",
  0: "#9baba4",
};
export const heroes = catalog.heroes;
export const items = catalog.items;
export const heroById = new Map(heroes.map((h) => [h.id, h]));
export const itemById = new Map(items.map((i) => [i.id, i]));
export const asset = (path: string) =>
  import.meta.env.BASE_URL + path.replace(/^\//, "");
export const number = (node: RawNode, key: string, fallback = 0): number =>
  typeof node[key] === "number" ? (node[key] as number) : fallback;
export const clock = (seconds: number) =>
  `${Math.floor(Math.max(0, seconds) / 60)}:${Math.floor(
    Math.max(0, seconds) % 60,
  )
    .toString()
    .padStart(2, "0")}`;
export const laneName = (lane: RawValue | undefined) =>
  ({ 1: "York", 4: "Broadway", 6: "Greenwich" })[String(lane)] ??
  `Lane ${lane ?? "unknown"}`;
export const objectiveName = (node: RawNode) =>
  String(node.objective_type ?? "Objective")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
export interface Scenario {
  format_version: 1;
  name: string;
  graph: RawGraph;
}
export type Selection =
  | { kind: "players" | "objectives" | "ziplines"; index: number }
  | { kind: "world" };
export const presets = initialPresets as (Scenario & { description: string })[];
export const clone = <T>(value: T): T => structuredClone(value);
export function initialScenario(): Scenario {
  try {
    const saved = localStorage.getItem("dwp-scenario-v1");
    if (saved) return parseScenario(saved);
  } catch {
    /* Start from a preset if a saved draft is unavailable. */
  }
  return clone(presets[2]);
}

const keys = new Set([
  "elapsed_seconds",
  "mid_boss_up",
  "next_midboss",
  "hero_id",
  "team_num",
  "x",
  "y",
  "z",
  "health",
  "max_health",
  "barrier",
  "level",
  "souls",
  "spent_souls",
  "respawn_time",
  "is_alive",
  "health_regen",
  "has_rebirth",
  "has_rejuvenator",
  "has_ultimate_trained",
  "ultimate_cooldown_start",
  "ultimate_cooldown_end",
  "ap_net_worth",
  "gold_net_worth",
  "upgrade_ids",
  "item_ids",
  "objective_type",
  "lane",
  "phase",
  "destroyed",
  "capturable",
  "cumulative_distance",
  "node_handle",
  "connections",
]);
/** Validate user files before they can allocate tensors; preserve missing values instead of inventing zeroes. */
export function parseScenario(text: string): Scenario {
  if (text.length > 2_000_000)
    throw new Error("Choose a scenario JSON smaller than 2 MB.");
  const value = JSON.parse(text);
  if (
    value?.format_version !== 1 ||
    typeof value.name !== "string" ||
    !value.name.trim() ||
    value.name.length > 100
  )
    throw new Error("This is not a supported scenario file.");
  const g = value.graph;
  const node = (input: unknown): RawNode => {
    if (!input || typeof input !== "object" || Array.isArray(input))
      throw new Error("Each entity must be an object.");
    const out: RawNode = {};
    for (const [key, v] of Object.entries(input)) {
      if (!keys.has(key)) continue;
      if (
        v !== null &&
        typeof v !== "boolean" &&
        !(typeof v === "string" && v.length <= 100) &&
        !(
          typeof v === "number" &&
          Number.isFinite(v) &&
          Math.abs(v) <= 2 ** 40
        ) &&
        !(
          Array.isArray(v) &&
          v.length <= 128 &&
          v.every((x) => Number.isSafeInteger(x) && x >= 0 && x <= 2 ** 40)
        )
      )
        throw new Error(`Invalid value for ${key}.`);
      out[key] = v as RawValue;
    }
    for (const key of [
      "x",
      "y",
      "z",
      "elapsed_seconds",
      "next_midboss",
      "ultimate_cooldown_start",
      "ultimate_cooldown_end",
      "cumulative_distance",
      "hero_id",
      "team_num",
      "node_handle",
    ])
      if (out[key] != null && typeof out[key] !== "number")
        throw new Error(`${key} must be a number or missing.`);
    for (const key of [
      "mid_boss_up",
      "is_alive",
      "has_rebirth",
      "has_rejuvenator",
      "has_ultimate_trained",
      "destroyed",
      "capturable",
    ])
      if (out[key] != null && typeof out[key] !== "boolean")
        throw new Error(`${key} must be true, false or missing.`);
    for (const key of ["upgrade_ids", "item_ids", "connections"])
      if (out[key] != null && !Array.isArray(out[key]))
        throw new Error(`${key} must be a list or missing.`);
    for (const key of [
      "health",
      "max_health",
      "barrier",
      "souls",
      "spent_souls",
      "respawn_time",
      "level",
      "gold_net_worth",
      "ap_net_worth",
      "health_regen",
    ])
      if (out[key] != null && (typeof out[key] !== "number" || out[key] < 0))
        throw new Error(`${key} must be a nonnegative number or missing.`);
    if (
      typeof out.health === "number" &&
      typeof out.max_health === "number" &&
      out.health > out.max_health
    )
      throw new Error("Health cannot exceed maximum health.");
    return out;
  };
  if (
    !g ||
    !Array.isArray(g.players) ||
    !g.players.length ||
    g.players.length > 12 ||
    !Array.isArray(g.objectives) ||
    g.objectives.length > 128 ||
    !Array.isArray(g.ziplines) ||
    g.ziplines.length > 512
  )
    throw new Error(
      "A scenario supports 1–12 heroes, up to 128 objectives and 512 zipline points.",
    );
  const players = g.players.map(node),
    objectives = g.objectives.map(node),
    ziplines = g.ziplines.map(node);
  const ids = new Set<number>();
  for (const p of players) {
    if (
      !heroById.has(number(p, "hero_id")) ||
      !TEAMS.includes(number(p, "team_num") as 2 | 3) ||
      ids.has(number(p, "hero_id"))
    )
      throw new Error("Each player needs a unique hero and a valid team.");
    ids.add(number(p, "hero_id"));
    if (
      p.upgrade_ids != null &&
      (!Array.isArray(p.upgrade_ids) ||
        new Set(p.upgrade_ids).size !== p.upgrade_ids.length)
    )
      throw new Error("Upgrade IDs must be a list of unique numbers.");
  }
  if (
    TEAMS.some(
      (t) => players.filter((p: RawNode) => p.team_num === t).length > 6,
    )
  )
    throw new Error("Each team can have up to six heroes.");
  const handles = new Set(ziplines.map((z: RawNode) => z.node_handle));
  if (
    handles.size !== ziplines.length ||
    ziplines.some(
      (z: RawNode) =>
        typeof z.node_handle !== "number" ||
        !Number.isSafeInteger(z.node_handle),
    )
  )
    throw new Error("Zipline point handles must be unique integers.");
  for (const z of ziplines)
    if (
      z.connections != null &&
      (!Array.isArray(z.connections) ||
        z.connections.some((h: number) => !handles.has(h)))
    )
      throw new Error("Zipline connections must refer to existing points.");
  const global_info = node(g.global_info);
  if (
    typeof global_info.elapsed_seconds !== "number" ||
    global_info.elapsed_seconds < 0 ||
    global_info.elapsed_seconds > 10800
  )
    throw new Error("Match time must be between 0 and 180 minutes.");
  return {
    format_version: 1,
    name: value.name,
    graph: {
      tick: Math.round(global_info.elapsed_seconds * 64),
      elapsed_seconds: global_info.elapsed_seconds,
      global_info,
      players,
      objectives,
      ziplines,
    },
  };
}
/** Inventory embeddings use every upgrade token; item selection must retain trained abilities. */
export function setInventory(player: RawNode, ids: number[]): RawNode {
  const abilities = Array.isArray(player.upgrade_ids)
    ? player.upgrade_ids.filter((id) => !itemById.has(id))
    : [];
  return {
    ...player,
    upgrade_ids: [...new Set([...abilities, ...ids])],
    item_ids: null,
  };
}
export function setHealth(player: RawNode, health: number | null): RawNode {
  return {
    ...player,
    health,
    ...(health === null
      ? {}
      : {
          is_alive: health > 0,
          respawn_time: health > 0 ? 0 : number(player, "respawn_time") || 20,
        }),
  };
}
export function setTime(scenario: Scenario, seconds: number): Scenario {
  const old = scenario.graph.elapsed_seconds,
    delta = seconds - old;
  const shift = (value: RawValue | undefined) =>
    typeof value === "number" && value > 0
      ? Math.max(0, value + delta)
      : (value ?? null);
  return {
    ...scenario,
    graph: {
      ...scenario.graph,
      tick: Math.round(seconds * 64),
      elapsed_seconds: seconds,
      global_info: {
        ...scenario.graph.global_info,
        elapsed_seconds: seconds,
        next_midboss: shift(scenario.graph.global_info.next_midboss),
      },
      players: scenario.graph.players.map((p) => ({
        ...p,
        ultimate_cooldown_start: shift(p.ultimate_cooldown_start),
        ultimate_cooldown_end: shift(p.ultimate_cooldown_end),
      })),
    },
  };
}
export const project = (x: number, y: number) =>
  [(x + 10752) / 21, (10752 - y) / 21] as const;
export const unproject = (x: number, y: number) =>
  [
    Math.max(-10752, Math.min(10752, x * 21 - 10752)),
    Math.max(-10752, Math.min(10752, 10752 - y * 21)),
  ] as const;
