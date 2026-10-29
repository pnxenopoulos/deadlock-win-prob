import { Coins, Plus, Skull, Users } from "lucide-react";
import type { RawGraph } from "../ml/types";
import {
  asset,
  heroById,
  number,
  TEAMS,
  TEAM_COLORS,
  TEAM_NAMES,
  type Selection,
} from "../scenario";
export function Roster({
  graph,
  selection,
  onSelect,
  onAdd,
}: {
  graph: RawGraph;
  selection: Selection;
  onSelect: (s: Selection) => void;
  onAdd: (team: number) => void;
}) {
  return (
    <div className="teams">
      {TEAMS.map((team) => {
        const players = graph.players
          .map((p, i) => ({ p, i }))
          .filter(({ p }) => p.team_num === team);
        const worth = players.reduce(
          (sum, { p }) => sum + number(p, "gold_net_worth"),
          0,
        );
        return (
          <section
            className="team"
            key={team}
            style={{ "--team": TEAM_COLORS[team] } as React.CSSProperties}
          >
            <div className="team-heading">
              <h2>
                <Users size={13} />
                {TEAM_NAMES[team]}
              </h2>
              <span title="Combined gold net worth">
                <Coins size={12} />
                {(worth / 1000).toFixed(1)}k
              </span>
            </div>
            <div className="team-list">
              {players.map(({ p, i }) => {
                const hero = heroById.get(number(p, "hero_id"));
                const alive =
                  p.is_alive !== false &&
                  (p.health == null || number(p, "health") > 0);
                return (
                  <button
                    key={i}
                    className={
                      "hero-row " +
                      (selection.kind === "players" && selection.index === i
                        ? "selected"
                        : "")
                    }
                    onClick={() => onSelect({ kind: "players", index: i })}
                    aria-label={`Edit ${hero?.name ?? "hero"}, ${TEAM_NAMES[team]}`}
                    aria-pressed={
                      selection.kind === "players" && selection.index === i
                    }
                  >
                    <span className={"portrait " + (!alive ? "dead" : "")}>
                      <img src={asset(hero?.icon ?? "")} alt="" />
                      {!alive && <Skull size={14} />}
                    </span>
                    <span className="hero-summary">
                      <span className="hero-title">
                        {hero?.name ?? "Unknown hero"}
                        <small>
                          {(number(p, "gold_net_worth") / 1000).toFixed(1)}k
                        </small>
                      </span>
                      <span className="health-track">
                        <span
                          style={{
                            width: `${Math.min(100, (100 * number(p, "health")) / Math.max(1, number(p, "max_health")))}%`,
                          }}
                        />
                      </span>
                      <span className="hero-health">
                        {alive
                          ? `${p.health == null ? "—" : Math.round(number(p, "health"))} / ${p.max_health == null ? "—" : Math.round(number(p, "max_health"))} HP`
                          : `Respawning · ${Math.round(number(p, "respawn_time"))}s`}
                        {number(p, "barrier") > 0 && (
                          <span>
                            {" "}
                            +{Math.round(number(p, "barrier"))} barrier
                          </span>
                        )}
                      </span>
                    </span>
                  </button>
                );
              })}
            </div>
            {players.length < 6 && (
              <button className="add-hero" onClick={() => onAdd(team)}>
                <Plus size={12} />
                Add hero
              </button>
            )}
          </section>
        );
      })}
    </div>
  );
}
