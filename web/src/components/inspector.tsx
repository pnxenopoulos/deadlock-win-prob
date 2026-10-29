import { useState } from "react";
import {
  Castle,
  Clock3,
  Crosshair,
  Heart,
  Package,
  Search,
  Shield,
  Skull,
  SlidersHorizontal,
  Waypoints,
  X,
} from "lucide-react";
import type { RawGraph, RawNode } from "../ml/types";
import {
  asset,
  clock,
  heroById,
  heroes,
  items,
  itemById,
  laneName,
  number,
  objectiveName,
  setHealth,
  setInventory,
  TEAM_COLORS,
  TEAM_NAMES,
  type Selection,
} from "../scenario";
import { Field, NumberField, Toggle } from "./fields";
interface Props {
  graph: RawGraph;
  selection: Selection;
  heroIds?: string[];
  upgradeIds?: string[];
  onPatch: (
    kind: "players" | "objectives" | "ziplines",
    index: number,
    patch: RawNode,
  ) => void;
  onWorld: (patch: RawNode) => void;
  onTime: (seconds: number) => void;
  onLane: (lane: unknown, team: number) => void;
}
export function Inspector({
  graph,
  selection,
  onPatch,
  onWorld,
  onTime,
  onLane,
  heroIds,
  upgradeIds,
}: Props) {
  const [query, setQuery] = useState("");
  const kind = selection.kind,
    node = kind === "world" ? graph.global_info : graph[kind][selection.index];
  if (!node) return null;
  const patch = (values: RawNode) => {
    if (selection.kind === "world") onWorld(values);
    else onPatch(selection.kind, selection.index, values);
  };
  const hero =
    kind === "players" ? heroById.get(number(node, "hero_id")) : undefined;
  const itemIds = Array.isArray(node.upgrade_ids)
    ? node.upgrade_ids.filter((id) => itemById.has(id))
    : [];
  const changeItem = (id: number) =>
    patch(
      setInventory(
        node,
        itemIds.includes(id)
          ? itemIds.filter((i) => i !== id)
          : [...itemIds, id],
      ),
    );
  const time = graph.elapsed_seconds;
  return (
    <aside className="inspector panel" aria-label="Scenario inspector">
      <div className="inspector-heading">
        <span className="eyebrow">
          <SlidersHorizontal size={13} />
          INSPECTOR
        </span>
        <span className="inspector-type">
          {kind === "players"
            ? "Hero"
            : kind === "objectives"
              ? "Objective"
              : kind === "ziplines"
                ? "Map control"
                : "World"}
        </span>
      </div>
      <div className="inspector-title">
        {hero ? (
          <img src={asset(hero.icon)} alt="" />
        ) : kind === "world" ? (
          <Clock3 />
        ) : kind === "ziplines" ? (
          <Waypoints />
        ) : (
          <Castle />
        )}
        <div>
          <h2>
            {hero?.name ??
              (kind === "world"
                ? "Match state"
                : kind === "ziplines"
                  ? `Zipline point ${selection.index + 1}`
                  : objectiveName(node))}
          </h2>
          <p style={{ color: TEAM_COLORS[number(node, "team_num")] }}>
            {kind === "world"
              ? clock(time) + " elapsed"
              : kind === "ziplines"
                ? laneName(node.lane)
                : (TEAM_NAMES[number(node, "team_num")] ?? "Neutral")}
          </p>
        </div>
      </div>
      <div className="inspector-content">
        {kind === "players" ? (
          <>
            <Field label="Hero">
              <select
                value={number(node, "hero_id")}
                onChange={(e) => {
                  const id = +e.target.value;
                  patch({
                    hero_id: id,
                    upgrade_ids: itemIds,
                    has_ultimate_trained: false,
                    ultimate_cooldown_start: 0,
                    ultimate_cooldown_end: 0,
                  });
                }}
              >
                {heroes.map((h) => (
                  <option
                    key={h.id}
                    value={h.id}
                    disabled={
                      (heroIds !== undefined &&
                        !heroIds.includes(String(h.id))) ||
                      graph.players.some(
                        (p) => p !== node && p.hero_id === h.id,
                      )
                    }
                  >
                    {h.name}
                    {heroIds && !heroIds.includes(String(h.id))
                      ? " · not in model"
                      : ""}
                  </option>
                ))}
              </select>
            </Field>
            <div className="section-label">
              <Heart size={13} />
              Vitality
            </div>
            <Toggle
              label="Alive"
              checked={node.is_alive !== false && number(node, "health") > 0}
              onChange={(alive) =>
                patch(
                  setHealth(node, alive ? number(node, "max_health", 650) : 0),
                )
              }
            />
            <div className="field-grid">
              <NumberField
                label="Health"
                value={node.health}
                min={0}
                max={number(node, "max_health", Infinity)}
                onChange={(health) => patch(setHealth(node, health))}
              />
              <NumberField
                label="Max health"
                value={node.max_health}
                min={1}
                onChange={(max_health) =>
                  patch({
                    max_health,
                    ...(max_health !== null &&
                    number(node, "health") > max_health
                      ? { health: max_health }
                      : {}),
                  })
                }
              />
              <NumberField
                label="Barrier"
                value={node.barrier}
                min={0}
                onChange={(barrier) => patch({ barrier })}
              />
              <NumberField
                label="Respawn timer"
                value={node.respawn_time}
                min={0}
                max={300}
                unit="s"
                onChange={(respawn_time) =>
                  patch({
                    respawn_time,
                    ...(respawn_time !== null && respawn_time > 0
                      ? { is_alive: false, health: 0 }
                      : {}),
                  })
                }
              />
            </div>
            <div className="section-label">
              <Crosshair size={13} />
              Progression
            </div>
            <div className="field-grid">
              <NumberField
                label="Level"
                value={node.level}
                min={0}
                max={100}
                onChange={(level) => patch({ level })}
              />
              <NumberField
                label="Net worth"
                value={node.gold_net_worth}
                min={0}
                onChange={(gold_net_worth) => patch({ gold_net_worth })}
              />
              <NumberField
                label="Unspent souls"
                value={node.souls}
                min={0}
                onChange={(souls) => patch({ souls })}
              />
              <NumberField
                label="Spent souls"
                value={node.spent_souls}
                min={0}
                onChange={(spent_souls) => patch({ spent_souls })}
              />
            </div>
            <div className="section-label">
              <Package size={13} />
              Items<span>{itemIds.length}</span>
            </div>
            <div className="inventory">
              {itemIds.map((id) => {
                const item = itemById.get(id)!;
                return (
                  <button
                    key={id}
                    aria-label={`Remove ${item.name}`}
                    title={`Remove ${item.name}`}
                    onClick={() => changeItem(id)}
                  >
                    <img src={asset(item.icon)} alt={item.name} />
                    <X size={10} />
                  </button>
                );
              })}
              {!itemIds.length && <p>No catalog items equipped</p>}
            </div>
            <details className="item-picker">
              <summary>Add or change items</summary>
              <label className="search">
                <Search size={14} />
                <input
                  aria-label="Search items"
                  placeholder="Search items…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </label>
              <div className="item-results">
                {items
                  .filter((i) =>
                    i.name.toLowerCase().includes(query.toLowerCase()),
                  )
                  .map((item) => (
                    <button
                      key={item.id}
                      title={item.name}
                      aria-label={`${itemIds.includes(item.id) ? "Remove" : "Add"} ${item.name}`}
                      aria-pressed={itemIds.includes(item.id)}
                      disabled={
                        upgradeIds !== undefined &&
                        !upgradeIds.includes(String(item.id))
                      }
                      onClick={() => changeItem(item.id)}
                    >
                      <img src={asset(item.icon)} alt="" />
                      <span>{item.name}</span>
                    </button>
                  ))}
              </div>
              <p className="help">
                Dimmed items were not represented in this model’s training
                vocabulary.
              </p>
            </details>
            <p className="help">
              Adjust health and economy separately when changing items. Hero
              swaps keep stats and items, and clear learned ability upgrades.
            </p>
            <div className="section-label">
              <Shield size={13} />
              Buffs & abilities
            </div>
            <Toggle
              label="Rejuvenator"
              checked={node.has_rejuvenator === true}
              onChange={(has_rejuvenator) => patch({ has_rejuvenator })}
            />
            <Toggle
              label="Rebirth available"
              checked={node.has_rebirth === true}
              onChange={(has_rebirth) => patch({ has_rebirth })}
            />
            <Toggle
              label="Ultimate trained"
              checked={node.has_ultimate_trained === true}
              onChange={(has_ultimate_trained) =>
                patch({ has_ultimate_trained })
              }
            />
            <details className="advanced">
              <summary>Advanced values</summary>
              <div className="field-grid">
                <NumberField
                  label="Height (Z)"
                  value={node.z}
                  step={10}
                  onChange={(z) => patch({ z })}
                />
                <NumberField
                  label="Health regeneration"
                  value={node.health_regen}
                  min={0}
                  onChange={(health_regen) => patch({ health_regen })}
                />
                <NumberField
                  label="Ability point worth"
                  value={node.ap_net_worth}
                  min={0}
                  onChange={(ap_net_worth) => patch({ ap_net_worth })}
                />
                <NumberField
                  label="Ultimate cooldown start"
                  value={node.ultimate_cooldown_start}
                  min={0}
                  unit="s"
                  onChange={(ultimate_cooldown_start) =>
                    patch({ ultimate_cooldown_start })
                  }
                />
                <NumberField
                  label="Ultimate cooldown end"
                  value={node.ultimate_cooldown_end}
                  min={0}
                  unit="s"
                  onChange={(ultimate_cooldown_end) =>
                    patch({ ultimate_cooldown_end })
                  }
                />
              </div>
              <p className="help">
                Cooldown timestamps use the match clock. Empty numeric fields
                remain unknown to the model.
              </p>
            </details>
          </>
        ) : kind === "objectives" ? (
          <>
            <div className="section-label">
              <Heart size={13} />
              Objective state
            </div>
            <Toggle
              label="Destroyed"
              checked={node.destroyed === true}
              onChange={(destroyed) =>
                patch({
                  destroyed,
                  health: destroyed ? 0 : number(node, "max_health"),
                })
              }
            />
            <NumberField
              label="Objective health"
              value={node.health}
              min={0}
              max={number(node, "max_health", Infinity)}
              onChange={(health) => patch({ health, destroyed: health === 0 })}
            />
            <NumberField
              label="Objective max health"
              value={node.max_health}
              min={1}
              onChange={(max_health) =>
                patch({
                  max_health,
                  ...(max_health !== null && number(node, "health") > max_health
                    ? { health: max_health }
                    : {}),
                })
              }
            />
            <Field label="Objective team">
              <select
                value={number(node, "team_num")}
                onChange={(e) => patch({ team_num: +e.target.value })}
              >
                {[3, 2, 0].map((team) => (
                  <option value={team} key={team}>
                    {TEAM_NAMES[team]}
                  </option>
                ))}
              </select>
            </Field>
            <NumberField
              label="Objective phase"
              value={node.phase}
              min={0}
              max={2}
              onChange={(phase) => patch({ phase })}
            />
            <p className="help">
              Objective positions stay anchored to the map. Health and destroyed
              state update together.
            </p>
            <div className="section-label">All objectives</div>
            <div className="objective-list">
              {graph.objectives.map((o, i) => (
                <span key={i}>
                  <i
                    style={{
                      background:
                        TEAM_COLORS[number(o, "team_num")] ?? TEAM_COLORS[0],
                    }}
                  />
                  {objectiveName(o)}
                  <small>
                    {o.destroyed
                      ? "Destroyed"
                      : `${Math.round((number(o, "health") / Math.max(1, number(o, "max_health"))) * 100)}%`}
                  </small>
                </span>
              ))}
            </div>
          </>
        ) : kind === "ziplines" ? (
          <>
            <div className="section-label">
              <Waypoints size={13} />
              Point control
            </div>
            <Field label="Controlling team">
              <select
                disabled={!node.capturable}
                value={number(node, "team_num")}
                onChange={(e) => patch({ team_num: +e.target.value })}
              >
                {[3, 2, 0].map((team) => (
                  <option value={team} key={team}>
                    {TEAM_NAMES[team]}
                  </option>
                ))}
              </select>
            </Field>
            <p className="help">
              {node.capturable
                ? "Click map points to paint ownership. Fixed points retain their original team."
                : "This point cannot be captured; its ownership stays fixed."}
            </p>
            <div className="stat-pair">
              <span>Lane</span>
              <strong>{laneName(node.lane)}</strong>
            </div>
            <div className="stat-pair">
              <span>Connections</span>
              <strong>
                {Array.isArray(node.connections)
                  ? node.connections.length
                  : "Unknown"}
              </strong>
            </div>
            <button
              className="button full"
              disabled={!node.capturable}
              onClick={() => onLane(node.lane, number(node, "team_num"))}
            >
              Apply team to this lane
            </button>
            <p className="help">
              Only capturable points in this lane change. The map’s geometry and
              directed connections are preserved.
            </p>
          </>
        ) : (
          <>
            <div className="section-label">
              <Clock3 size={13} />
              Match clock
            </div>
            <div className="field-grid">
              <NumberField
                label="Minutes"
                value={Math.floor(time / 60)}
                min={0}
                max={179}
                onChange={(minutes) =>
                  onTime((minutes ?? 0) * 60 + Math.floor(time % 60))
                }
              />
              <NumberField
                label="Seconds"
                value={Math.floor(time % 60)}
                min={0}
                max={59}
                onChange={(seconds) =>
                  onTime(Math.floor(time / 60) * 60 + (seconds ?? 0))
                }
              />
            </div>
            <div className="section-label">
              <Skull size={13} />
              Mid Boss
            </div>
            <Field label="Mid Boss state">
              <select
                value={
                  node.mid_boss_up === null
                    ? "unknown"
                    : node.mid_boss_up
                      ? "alive"
                      : "down"
                }
                onChange={(e) =>
                  onWorld({
                    mid_boss_up:
                      e.target.value === "unknown"
                        ? null
                        : e.target.value === "alive",
                    next_midboss: e.target.value === "unknown" ? null : 0,
                  })
                }
              >
                <option value="alive">Alive</option>
                <option value="down">Down / respawning</option>
                <option value="unknown">Unknown</option>
              </select>
            </Field>
            {node.mid_boss_up === false && (
              <NumberField
                label="Mid Boss respawn timer"
                value={
                  node.next_midboss === null
                    ? null
                    : Math.max(0, number(node, "next_midboss") - time)
                }
                min={0}
                max={1800}
                unit="s"
                onChange={(remaining) =>
                  onWorld({
                    next_midboss: remaining === null ? null : time + remaining,
                  })
                }
              />
            )}
            <p className="help">
              Set Rejuvenator on the heroes who hold it in their individual
              settings.
            </p>
            <div className="section-label">Scenario overview</div>
            <div className="stat-pair">
              <span>Heroes</span>
              <strong>{graph.players.length}</strong>
            </div>
            <div className="stat-pair">
              <span>Objectives alive</span>
              <strong>
                {graph.objectives.filter((o) => !o.destroyed).length}
              </strong>
            </div>
            <div className="stat-pair">
              <span>Zipline points</span>
              <strong>{graph.ziplines.length}</strong>
            </div>
            <p className="help">
              Changing time preserves existing cooldown and respawn offsets. It
              does not simulate movement, farming, or combat.
            </p>
          </>
        )}
      </div>
    </aside>
  );
}
