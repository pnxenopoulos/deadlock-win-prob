import { expect, test } from "bun:test";
import {
  parseScenario,
  presets,
  clone,
  setInventory,
  setHealth,
  setTime,
  items,
  project,
  unproject,
} from "../src/scenario";
import { encodeGraphs, validateEncoder } from "../src/ml/features";
import type { Encoder, RawGraph } from "../src/ml/types";
import fixture from "./fixtures/encoding.json";

test("every identity-free preset survives import/export with its actual topology", () => {
  for (const preset of presets) {
    const saved = JSON.stringify(preset);
    const imported = parseScenario(saved);
    expect(imported.graph.players).toHaveLength(12);
    expect(imported.graph.ziplines).toEqual(preset.graph.ziplines);
    expect(imported.graph.players).toEqual(preset.graph.players);
    expect(saved).not.toContain("steam_id");
    expect(saved).not.toContain("entity_serial");
  }
});
test("imports reject invalid state before tensor allocation", () => {
  for (const edit of [
    (g: RawGraph) => {
      g.players = Array(13).fill(g.players[0]);
    },
    (g: RawGraph) => {
      g.players[0].health = -1;
    },
    (g: RawGraph) => {
      g.players[0].health = 10000000;
    },
    (g: RawGraph) => {
      g.players[1].hero_id = g.players[0].hero_id;
    },
    (g: RawGraph) => {
      g.ziplines[0].connections = [999999999];
    },
    (g: RawGraph) => {
      g.global_info.elapsed_seconds = -100;
    },
    (g: RawGraph) => {
      g.players[0].is_alive = "true";
    },
  ]) {
    const preset = clone(presets[2]);
    edit(preset.graph);
    expect(() => parseScenario(JSON.stringify(preset))).toThrow();
  }
  expect(() => parseScenario("x".repeat(2000001))).toThrow("2 MB");
});
test("unknown fields are stripped and missing numeric inputs remain missing", () => {
  const preset = clone(presets[2]);
  preset.graph.players[0].health_regen = null;
  preset.graph.players[0].steam_id = "private";
  const p = parseScenario(JSON.stringify(preset)).graph.players[0];
  expect(p.health_regen).toBeNull();
  expect(p.steam_id).toBeUndefined();
});
test("inventory edits retain ability tokens and avoid fabricating an observed item count", () => {
  const item = items[0].id;
  const player = { upgrade_ids: [12345, item], item_ids: null };
  expect(setInventory(player, [])).toEqual({
    upgrade_ids: [12345],
    item_ids: null,
  });
  expect(setInventory(player, [item, item]).upgrade_ids).toEqual([12345, item]);
});
test("health and respawn state stay consistent", () => {
  const dead = setHealth({ health: 100, is_alive: true, respawn_time: 0 }, 0);
  expect(dead.is_alive).toBe(false);
  expect(dead.respawn_time).toBe(20);
  const alive = setHealth(dead, 100);
  expect(alive.is_alive).toBe(true);
  expect(alive.respawn_time).toBe(0);
});
test("moving time preserves relative cooldowns and uses the model's absolute boss timer", () => {
  const s = clone(presets[2]);
  s.graph.players[0].ultimate_cooldown_end = s.graph.elapsed_seconds + 30;
  s.graph.global_info.next_midboss = s.graph.elapsed_seconds + 300;
  const next = setTime(s, 1200);
  expect(next.graph.players[0].ultimate_cooldown_end).toBe(1230);
  expect(next.graph.global_info.next_midboss).toBe(1500);
  expect(next.graph.global_info.elapsed_seconds).toBe(1200);
  expect(s.graph.global_info.elapsed_seconds).not.toBe(1200);
});
test("map transforms round trip world coordinates and clamp placements", () => {
  for (const [x, y] of [
    [0, 0],
    [-10000, 5000],
    [10752, -10752],
  ]) {
    const p = project(x, y);
    const result = unproject(...p);
    expect(result[0]).toBeCloseTo(x);
    expect(result[1]).toBeCloseTo(y);
  }
  expect(unproject(-100, 2000)).toEqual([-10752, -10752]);
});
test("the reused browser encoder matches Python across every feature tensor", () => {
  const encoder = fixture.encoder as Encoder;
  validateEncoder(encoder);
  const encoded = encodeGraphs(fixture.graphs as RawGraph[], encoder);
  for (const [name, input] of Object.entries(encoded)) {
    const expected = fixture.inputs[name as keyof typeof fixture.inputs];
    expect(input.dims).toEqual(expected.dims);
    expect(String(input.type)).toBe(expected.type);
    expect(input.data.length).toBe(expected.data.length);
    input.data.forEach((v, i) =>
      expect(
        Math.abs(Number(v) - Number(expected.data[i])),
      ).toBeLessThanOrEqual(input.type === "float32" ? 2e-6 : 0),
    );
  }
});
