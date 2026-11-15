import { useEffect, useRef, useState } from "react";
import type { ModelOption } from "./models";
import type { RawGraph } from "./types";
export interface Prediction {
  probability?: number;
  baseline?: number;
  modelVersion?: string;
  heroIds?: string[];
  upgradeIds?: string[];
  error?: string;
  pending: boolean;
}
export function usePrediction(graph: RawGraph, baseline: RawGraph | null, model: ModelOption) {
  const worker = useRef<Worker | null>(null),
    sequence = useRef(0);
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<Prediction & { modelId?: string }>({ pending: true });
  useEffect(() => {
    const w = new Worker(new URL("./predict.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.current = w;
    w.onmessage = ({ data }) => {
      if (worker.current !== w || data.id !== sequence.current) return;
      setState((previous) =>
        data.error
          ? { ...previous, error: data.error, pending: false }
          : {
              ...data,
              modelId: model.id,
              probability: data.probabilities[0],
              baseline: data.probabilities[1],
              pending: false,
            },
      );
    };
    w.onerror = () => {
      if (worker.current !== w) return;
      setState((previous) => ({
        ...previous,
        pending: false,
        modelId: model.id,
        error: "The prediction worker stopped. Please retry.",
      }));
    };
    return () => {
      w.terminate();
      worker.current = null;
    };
  }, [attempt, model.id, model.base, model.version]);
  useEffect(() => {
    const id = ++sequence.current;
    setState((previous) => previous.modelId === model.id
      ? { ...previous, pending: true, error: undefined }
      : { modelId: model.id, pending: true });
    const timer = setTimeout(
      () =>
        worker.current?.postMessage({
          id,
          graphs: baseline ? [graph, baseline] : [graph],
          model,
        }),
      180,
    );
    return () => clearTimeout(timer);
  }, [graph, baseline, attempt, model]);
  return { ...(state.modelId === model.id ? state : { pending: true }), retry: () => setAttempt((a) => a + 1) };
}
