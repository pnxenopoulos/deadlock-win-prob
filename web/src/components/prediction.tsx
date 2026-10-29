import { Activity, LoaderCircle, Pin, X } from "lucide-react";
import type { Prediction as Estimate } from "../ml/use-prediction";
export function Prediction({
  estimate,
  onPin,
  onClear,
  hasBaseline,
}: {
  estimate: Estimate & { retry: () => void };
  onPin: () => void;
  onClear: () => void;
  hasBaseline: boolean;
}) {
  const available = estimate.probability !== undefined && !estimate.error;
  const p = estimate.probability ?? 0.5;
  const delta =
    estimate.baseline !== undefined && available
      ? (p - estimate.baseline) * 100
      : null;
  return (
    <section className="prediction panel" aria-label="Win probability">
      <div className="eyebrow">
        <Activity size={13} />
        WIN PROBABILITY
        <span className={"status-dot " + (available ? "ready" : "")} />
      </div>
      {estimate.error ? (
        <div className="prediction-error" role="alert">
          <p>{estimate.error}</p>
          <button onClick={estimate.retry}>Retry prediction</button>
        </div>
      ) : (
        <>
          <div
            className={"probabilities " + (estimate.pending ? "updating" : "")}
            aria-live="polite"
          >
            <div className="amber">
              <strong>
                {available ? (p * 100).toFixed(1) : "—"}
                <span>%</span>
              </strong>
              <span>Archmother</span>
            </div>
            <span className="versus">/</span>
            <div className="blue">
              <strong>
                {available ? ((1 - p) * 100).toFixed(1) : "—"}
                <span>%</span>
              </strong>
              <span>Hidden King</span>
            </div>
          </div>
          <div className="probability-bar" aria-hidden>
            <span style={{ width: `${p * 100}%` }} />
            <span style={{ width: `${(1 - p) * 100}%` }} />
          </div>
          <div className="estimate-status" role="status">
            {estimate.pending ? (
              <>
                <LoaderCircle size={11} className="spin" />
                {available ? "Updating estimate…" : "Loading model…"}
              </>
            ) : (
              <>Current scenario</>
            )}
          </div>
        </>
      )}
      {hasBaseline ? (
        <div className="baseline">
          <span>
            {delta !== null
              ? `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pp`
              : "—"}
            <small> Archmother vs. baseline</small>
          </span>
          <button aria-label="Clear baseline" onClick={onClear}>
            <X size={13} />
          </button>
        </div>
      ) : (
        <button
          className="baseline-button"
          disabled={!available || estimate.pending}
          onClick={onPin}
        >
          <Pin size={12} />
          Set as baseline<span>Compare your changes</span>
        </button>
      )}
    </section>
  );
}
