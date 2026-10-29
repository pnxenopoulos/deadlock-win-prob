import { useRef, useState } from "react";
import {
  Castle,
  Crown,
  Flag,
  Gem,
  Maximize,
  Minus,
  Plus,
  Shield,
  ShieldHalf,
  Skull,
  Users,
  Waypoints,
} from "lucide-react";
import type { RawGraph, RawNode } from "../ml/types";
import {
  asset,
  heroById,
  number,
  objectiveName,
  project,
  unproject,
  TEAM_COLORS,
  TEAM_NAMES,
  type Selection,
} from "../scenario";
export type MapMode = "players" | "objectives" | "ziplines";
const icons: Record<string, typeof Shield> = {
  guardian: Shield,
  base_guardian: ShieldHalf,
  walker: Castle,
  patron: Crown,
  shrine: Gem,
  mid_boss: Skull,
};
export function ScenarioMap({
  graph,
  selection,
  onSelect,
  onMove,
  onPaint,
  mode,
  onMode,
}: {
  graph: RawGraph;
  selection: Selection;
  onSelect: (s: Selection) => void;
  onMove: (index: number, x: number, y: number) => void;
  onPaint: (index: number, team: number) => void;
  mode: MapMode;
  onMode: (mode: MapMode) => void;
}) {
  const [layer, setLayer] = useState("surface"),
    [zoom, setZoom] = useState(1),
    [brush, setBrush] = useState(3);
  const svg = useRef<SVGSVGElement>(null),
    dragging = useRef<number | null>(null),
    moved = useRef(false);
  const handles = new Map(graph.ziplines.map((z, i) => [z.node_handle, i]));
  const pointer = (e: { clientX: number; clientY: number }) => {
    const point = svg.current!.createSVGPoint();
    point.x = e.clientX;
    point.y = e.clientY;
    const p = point.matrixTransform(svg.current!.getScreenCTM()!.inverse());
    return unproject(p.x, p.y);
  };
  const size = 1024 / zoom;
  const dot = (node: RawNode) => project(number(node, "x"), number(node, "y"));
  return (
    <section className="map-stage panel" aria-label="Scenario map">
      <div className="map-header">
        <div>
          <h1>Scenario map</h1>
          <p>
            {mode === "players"
              ? "Drag a hero, or select one and click to reposition."
              : mode === "ziplines"
                ? "Choose a team, then click a point to change control."
                : "Select an objective to change its health or state."}
          </p>
        </div>
        <div className="layer-switch" aria-label="Map layer">
          {["surface", "tunnels"].map((value) => (
            <button
              key={value}
              aria-pressed={layer === value}
              onClick={() => setLayer(value)}
            >
              {value === "surface" ? "Surface" : "Tunnels"}
            </button>
          ))}
        </div>
      </div>
      <div className="map-tools" role="group" aria-label="Map editing mode">
        {(
          [
            { key: "players", label: "Heroes", Icon: Users },
            { key: "objectives", label: "Objectives", Icon: Castle },
            { key: "ziplines", label: "Ziplines", Icon: Waypoints },
          ] as const
        ).map(({ key, label, Icon }) => (
          <button
            key={key}
            aria-pressed={mode === key}
            onClick={() => onMode(key)}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
        {mode === "ziplines" && (
          <div className="paint-team">
            {[3, 2, 0].map((team) => (
              <button
                key={team}
                aria-label={`Paint ${TEAM_NAMES[team]}`}
                aria-pressed={brush === team}
                onClick={() => setBrush(team)}
                style={{ color: TEAM_COLORS[team] }}
              >
                <span style={{ background: TEAM_COLORS[team] }} />
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="map-canvas">
        <svg
          ref={svg}
          role="group"
          aria-label="Interactive Deadlock map"
          viewBox={`${512 - size / 2} ${512 - size / 2} ${size} ${size}`}
          onPointerMove={(e) => {
            if (dragging.current === null) return;
            moved.current = true;
            const [x, y] = pointer(e);
            onMove(dragging.current, x, y);
          }}
          onPointerUp={() => {
            dragging.current = null;
          }}
          onPointerCancel={() => {
            dragging.current = null;
          }}
          onClick={(e) => {
            if (moved.current) {
              moved.current = false;
              return;
            }
            if (
              e.target === e.currentTarget ||
              (e.target as Element).getAttribute("data-map-background")
            ) {
              if (mode === "players" && selection.kind === "players") {
                const [x, y] = pointer(e);
                onMove(selection.index, x, y);
              }
            }
          }}
        >
          <image
            data-map-background="true"
            href={asset(
              layer === "surface"
                ? "/minimap/minimap_midtown_mid_psd.webp"
                : "/minimap/minimap_midtown_mid_tunnels_psd.webp",
            )}
            width={1024}
            height={1024}
          />
          {mode === "ziplines" && (
            <g className="zipline-links" pointerEvents="none">
              {graph.ziplines.flatMap((z, i) =>
                Array.isArray(z.connections)
                  ? z.connections.map((handle) => {
                      const j = handles.get(handle);
                      if (j === undefined) return null;
                      const a = dot(z),
                        b = dot(graph.ziplines[j]);
                      return (
                        <line
                          key={`${i}-${j}`}
                          x1={a[0]}
                          y1={a[1]}
                          x2={b[0]}
                          y2={b[1]}
                          stroke={
                            TEAM_COLORS[number(z, "team_num")] ?? TEAM_COLORS[0]
                          }
                          strokeWidth={2.4 / zoom}
                          opacity={0.5}
                        />
                      );
                    })
                  : [],
              )}
            </g>
          )}
          <g>
            {graph.objectives.map((o, i) => {
              const [x, y] = dot(o),
                Icon = icons[String(o.objective_type)] ?? Flag;
              const selected =
                selection.kind === "objectives" && selection.index === i;
              return (
                <g
                  key={i}
                  transform={`translate(${x},${y}) scale(${1 / zoom})`}
                  role="button"
                  tabIndex={0}
                  aria-label={`${TEAM_NAMES[number(o, "team_num")] ?? "Neutral"} ${objectiveName(o)} ${i + 1}`}
                  className="map-objective"
                  style={{
                    color: TEAM_COLORS[number(o, "team_num")] ?? TEAM_COLORS[0],
                    opacity: o.destroyed ? 0.3 : mode === "ziplines" ? 0.4 : 1,
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelect({ kind: "objectives", index: i });
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect({ kind: "objectives", index: i });
                    }
                  }}
                >
                  <title>
                    {objectiveName(o)} · {Math.round(number(o, "health"))} HP
                  </title>
                  <circle
                    r={selected ? 17 : 12}
                    fill="#222021"
                    stroke="currentColor"
                    strokeWidth={selected ? 3 : 2}
                  />
                  <Icon x={-8} y={-8} width={16} height={16} />
                  {o.destroyed === true && (
                    <path
                      d="M-8 -8 8 8M8 -8 -8 8"
                      stroke="currentColor"
                      strokeWidth={2}
                    />
                  )}
                </g>
              );
            })}
          </g>
          {mode === "ziplines" && (
            <g>
              {graph.ziplines.map((z, i) => {
                const [x, y] = dot(z);
                const selected =
                  selection.kind === "ziplines" && selection.index === i;
                return (
                  <circle
                    key={i}
                    cx={x}
                    cy={y}
                    r={(selected ? 7 : 4) / zoom}
                    fill={TEAM_COLORS[number(z, "team_num")] ?? TEAM_COLORS[0]}
                    stroke={selected ? "#efdebf" : "#222021"}
                    strokeWidth={1.5 / zoom}
                    className="zip-point"
                    role="button"
                    tabIndex={0}
                    aria-label={`Zipline point ${i + 1}, ${TEAM_NAMES[number(z, "team_num")] ?? "Neutral"}${!z.capturable ? ", fixed" : ""}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelect({ kind: "ziplines", index: i });
                      if (z.capturable) onPaint(i, brush);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onSelect({ kind: "ziplines", index: i });
                        if (z.capturable) onPaint(i, brush);
                      }
                    }}
                  />
                );
              })}
            </g>
          )}
          <defs>
            {graph.players.map((_, i) => (
              <clipPath id={`hero-clip-${i}`} key={i}>
                <circle r={16} />
              </clipPath>
            ))}
          </defs>
          <g>
            {graph.players.map((p, i) => {
              const [x, y] = dot(p),
                hero = heroById.get(number(p, "hero_id"));
              const selected =
                selection.kind === "players" && selection.index === i;
              return (
                <g
                  key={i}
                  transform={`translate(${x},${y}) scale(${1.35 / zoom})`}
                  role="button"
                  tabIndex={0}
                  aria-label={`Move ${hero?.name ?? "hero"}`}
                  className="map-hero"
                  style={{
                    color: TEAM_COLORS[number(p, "team_num")],
                    opacity:
                      mode === "ziplines"
                        ? 0.4
                        : p.is_alive === false
                          ? 0.5
                          : 1,
                  }}
                  onPointerDown={(e) => {
                    if (mode !== "players") return;
                    e.stopPropagation();
                    onSelect({ kind: "players", index: i });
                    dragging.current = i;
                    moved.current = false;
                    svg.current!.setPointerCapture(e.pointerId);
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelect({ kind: "players", index: i });
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect({ kind: "players", index: i });
                    }
                    const delta: Record<string, number[]> = {
                      ArrowLeft: [-100, 0],
                      ArrowRight: [100, 0],
                      ArrowUp: [0, 100],
                      ArrowDown: [0, -100],
                    };
                    if (delta[e.key]) {
                      e.preventDefault();
                      onMove(
                        i,
                        number(p, "x") + delta[e.key][0],
                        number(p, "y") + delta[e.key][1],
                      );
                    }
                  }}
                >
                  <title>
                    {hero?.name} · {TEAM_NAMES[number(p, "team_num")]}
                  </title>
                  {selected && (
                    <circle
                      r={23}
                      fill="none"
                      stroke="#efdebf"
                      strokeWidth={1.5}
                      strokeDasharray="3 3"
                    />
                  )}
                  <circle
                    r={18}
                    fill="#222021"
                    stroke="currentColor"
                    strokeWidth={3}
                  />
                  <image
                    href={asset(hero?.icon ?? "")}
                    x={-16}
                    y={-16}
                    width={32}
                    height={32}
                    clipPath={`url(#hero-clip-${i})`}
                  />
                  {p.is_alive === false && (
                    <Skull
                      x={-8}
                      y={-8}
                      width={16}
                      height={16}
                      fill="#222021"
                      color="#efdebf"
                    />
                  )}
                </g>
              );
            })}
          </g>
        </svg>
        <div className="zoom-tools">
          <button
            aria-label="Zoom in"
            onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}
          >
            <Plus size={15} />
          </button>
          <button
            aria-label="Zoom out"
            onClick={() => setZoom((z) => Math.max(1, z - 0.25))}
            disabled={zoom <= 1}
          >
            <Minus size={15} />
          </button>
          <button aria-label="Reset zoom" onClick={() => setZoom(1)}>
            <Maximize size={14} />
          </button>
        </div>
      </div>
      <div className="map-caption">
        <span>
          <i className="amber-dot" />
          Archmother
        </span>
        <span>
          <i className="blue-dot" />
          Hidden King
        </span>
        <span>
          {mode === "ziplines"
            ? `${graph.ziplines.length} connected zipline points`
            : "Arrow keys move a focused hero"}
        </span>
      </div>
    </section>
  );
}
