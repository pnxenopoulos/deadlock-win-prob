import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  Clock3,
  Download,
  Github,
  Info,
  Map,
  Moon,
  RotateCcw,
  Sun,
  Upload,
  X,
} from "lucide-react";
import { Inspector } from "./components/inspector";
import { Prediction } from "./components/prediction";
import { Roster } from "./components/roster";
import { ScenarioMap, type MapMode } from "./components/scenario-map";
import { usePrediction } from "./ml/use-prediction";
import type { RawGraph, RawNode } from "./ml/types";
import {
  clone,
  clock,
  initialScenario,
  number,
  parseScenario,
  presets,
  setTime,
  type Scenario,
  type Selection,
} from "./scenario";

export function App() {
  const [scenario, setScenario] = useState(initialScenario),
    [preset, setPreset] = useState(() => {
      try {
        const saved = localStorage.getItem("dwp-scenario-v1");
        if (saved) {
          parseScenario(saved);
          return "custom";
        }
      } catch {}
      return "2";
    }),
    [selection, setSelection] = useState<Selection>({
      kind: "players",
      index: 0,
    });
  const [mode, setMode] = useState<MapMode>("players"),
    [baseline, setBaseline] = useState<RawGraph | null>(null),
    [notice, setNotice] = useState("");
  const [dark, setDark] = useState(() => {
    try {
      return localStorage.getItem("dwp-theme") !== "light";
    } catch {
      return true;
    }
  });
  const [info, setInfo] = useState(false),
    file = useRef<HTMLInputElement>(null),
    dialog = useRef<HTMLDialogElement>(null);
  const estimate = usePrediction(scenario.graph, baseline),
    graph = scenario.graph;
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    try {
      localStorage.setItem("dwp-theme", dark ? "dark" : "light");
    } catch {}
  }, [dark]);
  useEffect(() => {
    const timer = setTimeout(() => {
      try {
        localStorage.setItem(
          "dwp-scenario-v1",
          JSON.stringify({
            ...scenario,
            name: scenario.name.trim() || "Untitled scenario",
          }),
        );
      } catch {}
    }, 300);
    return () => clearTimeout(timer);
  }, [scenario]);
  useEffect(() => {
    if (info) dialog.current?.showModal();
    else dialog.current?.close();
  }, [info]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 5500);
    return () => clearTimeout(timer);
  }, [notice]);
  const patch = (
    kind: "players" | "objectives" | "ziplines",
    index: number,
    values: RawNode,
  ) =>
    setScenario((s) => ({
      ...s,
      graph: {
        ...s.graph,
        [kind]: s.graph[kind].map((node, i) =>
          i === index ? { ...node, ...values } : node,
        ),
      },
    }));
  const select = (value: Selection) => {
    setSelection(value);
    if (value.kind !== "world") setMode(value.kind);
  };
  const loadPreset = (id: string) => {
    setScenario(clone(presets[+id]));
    setPreset(id);
    setSelection({ kind: "players", index: 0 });
    setMode("players");
    setNotice("Preset loaded");
  };
  const exportScenario = () => {
    const payload = {
      ...scenario,
      name: scenario.name.trim() || "Untitled scenario",
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download =
      (payload.name.toLowerCase().replace(/[^a-z0-9]+/g, "-") || "scenario") +
      ".json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("Scenario exported");
  };
  return (
    <>
      <header className="site-header">
        <a className="brand" href={import.meta.env.BASE_URL}>
          <Map size={21} />
          <span>
            deadlock<span className="brand-light"> / scenarios</span>
          </span>
        </a>
        <span className="header-tag">Win probability sandbox</span>
        <nav>
          <a href="https://deadlock.nyc" target="_blank" rel="noreferrer">
            Replay viewer
            <ArrowUpRight size={12} />
          </a>
          <button
            aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
            onClick={() => setDark((d) => !d)}
          >
            {dark ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <button aria-label="About this model" onClick={() => setInfo(true)}>
            <Info size={17} />
          </button>
          <a
            href="https://github.com/pnxenopoulos/deadlock-win-prob"
            aria-label="GitHub"
            target="_blank"
            rel="noreferrer"
          >
            <Github size={17} />
          </a>
        </nav>
      </header>
      <main>
        <div className="scenario-toolbar">
          <div className="scenario-identity">
            <span className="eyebrow">SCENARIO</span>
            <input
              aria-label="Scenario name"
              value={scenario.name}
              maxLength={100}
              onChange={(e) =>
                setScenario((s) => ({ ...s, name: e.target.value }))
              }
              onBlur={() => {
                if (!scenario.name.trim())
                  setScenario((s) => ({ ...s, name: "Untitled scenario" }));
              }}
            />
          </div>
          <label className="preset-select">
            <span>Start from</span>
            <select
              aria-label="Scenario preset"
              value={preset}
              onChange={(e) => loadPreset(e.target.value)}
            >
              {preset === "custom" && (
                <option value="custom">Custom scenario</option>
              )}
              {presets.map((p, i) => (
                <option key={i} value={i}>
                  {p.name} · {clock(p.graph.elapsed_seconds)}
                </option>
              ))}
            </select>
          </label>
          <div className="toolbar-actions">
            <button
              className="button"
              onClick={() => select({ kind: "world" })}
            >
              <Clock3 size={14} />
              {clock(graph.elapsed_seconds)}
            </button>
            <button
              className="button icon-button"
              aria-label="Reset scenario"
              title="Reset to preset"
              onClick={() => loadPreset(preset === "custom" ? "2" : preset)}
            >
              <RotateCcw size={15} />
            </button>
            <button className="button" onClick={() => file.current?.click()}>
              <Upload size={14} />
              Import
            </button>
            <button className="button primary" onClick={exportScenario}>
              <Download size={14} />
              Export
            </button>
          </div>
        </div>
        <input
          type="file"
          ref={file}
          accept=".json,application/json"
          hidden
          aria-label="Import scenario JSON"
          onChange={async (e) => {
            const selected = e.target.files?.[0];
            e.target.value = "";
            if (!selected) return;
            try {
              if (selected.size > 2_000_000)
                throw new Error("Choose a scenario JSON smaller than 2 MB.");
              const imported = parseScenario(await selected.text());
              setScenario(imported);
              setPreset("custom");
              setSelection({ kind: "players", index: 0 });
              setNotice("Scenario imported");
            } catch (error) {
              setNotice(
                "Import failed: " +
                  (error instanceof Error ? error.message : String(error)),
              );
            }
          }}
        />
        <div className="workspace">
          <aside className="roster-pane">
            <Prediction
              estimate={estimate}
              hasBaseline={baseline !== null}
              onPin={() => {
                setBaseline(clone(graph));
                setNotice("Baseline saved. Edit the scenario to compare.");
              }}
              onClear={() => setBaseline(null)}
            />
            <Roster
              graph={graph}
              selection={selection}
              onSelect={select}
              onAdd={(team) => {
                const used = new Set(graph.players.map((p) => p.hero_id));
                const player = presets[0].graph.players.find(
                  (p) =>
                    !used.has(p.hero_id) &&
                    (!estimate.heroIds ||
                      estimate.heroIds.includes(String(p.hero_id))),
                );
                if (!player) {
                  setNotice(
                    "All available starting heroes are already on the map.",
                  );
                  return;
                }
                const index = graph.players.length;
                setScenario((s) => ({
                  ...s,
                  graph: {
                    ...s.graph,
                    players: [
                      ...s.graph.players,
                      { ...clone(player), team_num: team },
                    ],
                  },
                }));
                select({ kind: "players", index });
              }}
            />
          </aside>
          <div className="map-column">
            <ScenarioMap
              graph={graph}
              selection={selection}
              onSelect={select}
              mode={mode}
              onMode={setMode}
              onMove={(index, x, y) =>
                patch("players", index, {
                  x: Math.round(Math.max(-10752, Math.min(10752, x))),
                  y: Math.round(Math.max(-10752, Math.min(10752, y))),
                })
              }
              onPaint={(index, team_num) =>
                patch("ziplines", index, { team_num })
              }
            />
            <div className="time-control panel">
              <button onClick={() => select({ kind: "world" })}>
                <Clock3 size={15} />
                <span>
                  Match time<strong>{clock(graph.elapsed_seconds)}</strong>
                </span>
              </button>
              <input
                type="range"
                aria-label="Match time in seconds"
                min={0}
                max={Math.max(3600, graph.elapsed_seconds)}
                step={15}
                value={graph.elapsed_seconds}
                onChange={(e) =>
                  setScenario((s) => setTime(s, +e.target.value))
                }
              />
              <span className="time-end">
                {clock(Math.max(3600, graph.elapsed_seconds))}
              </span>
            </div>
            <p className="scenario-note">
              Estimates are most useful for realistic match states. All
              predictions run on your device.
            </p>
          </div>
          <Inspector
            graph={graph}
            selection={selection}
            heroIds={estimate.heroIds}
            upgradeIds={estimate.upgradeIds}
            onPatch={patch}
            onTime={(seconds) => setScenario((s) => setTime(s, seconds))}
            onWorld={(values) =>
              setScenario((s) => ({
                ...s,
                graph: {
                  ...s.graph,
                  global_info: { ...s.graph.global_info, ...values },
                },
              }))
            }
            onLane={(lane, team_num) =>
              setScenario((s) => ({
                ...s,
                graph: {
                  ...s.graph,
                  ziplines: s.graph.ziplines.map((z) =>
                    z.lane === lane && z.capturable ? { ...z, team_num } : z,
                  ),
                },
              }))
            }
          />
        </div>
      </main>
      {notice && (
        <div className="toast" role="status">
          {notice}
          <button
            aria-label="Dismiss notification"
            onClick={() => setNotice("")}
          >
            <X size={13} />
          </button>
        </div>
      )}
      <dialog
        ref={dialog}
        className="info-dialog"
        onCancel={() => setInfo(false)}
        onClick={(e) => {
          if (e.target === e.currentTarget) setInfo(false);
        }}
      >
        <div>
          <div className="modal-heading">
            <h2>About Deadlock Scenarios</h2>
            <button
              aria-label="Close information"
              autoFocus
              onClick={() => setInfo(false)}
            >
              <X size={18} />
            </button>
          </div>
          <p>
            Explore match states with{" "}
            <a
              href="https://github.com/pnxenopoulos/deadlock-win-prob"
              target="_blank"
              rel="noreferrer"
            >
              Deadlock Win Probability
            </a>
            .
          </p>
          <dl>
            <dt>Model</dt>
            <dd>
              {estimate.modelVersion ??
                (estimate.error ? "Unavailable" : "Loading…")}
            </dd>
            <dt>Map & catalog</dt>
            <dd>Midtown · boon-data 6759</dd>
          </dl>
          <p>
            The model estimates outcomes from recorded match states. Edited
            scenarios may differ from the states it learned from; a probability
            change does not establish a causal effect.
          </p>
          <p>
            Presets contain game state with player identities removed. Your
            scenarios stay in this browser unless you export them. Visuals and
            browser inference are adapted from{" "}
            <a href="https://deadlock.nyc" target="_blank" rel="noreferrer">
              deadlock.nyc
            </a>
            .
          </p>
          <p className="legal">
            Deadlock and related game content belong to Valve Corporation. This
            unofficial fan project is not affiliated with or endorsed by Valve.{" "}
            <a href={import.meta.env.BASE_URL + "NOTICE.txt"}>Asset notice</a>
          </p>
        </div>
      </dialog>
    </>
  );
}
