import { useEffect, useRef, useState } from "react";
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
export function usePrediction(graph: RawGraph, baseline: RawGraph | null) {
  const worker = useRef<Worker | null>(null),
    sequence = useRef(0);
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<Prediction>({ pending: true });
  useEffect(() => {
    const w = new Worker(new URL("./predict.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.current = w;
    w.onmessage = ({ data }) => {
      if (data.id !== sequence.current) return;
      setState((previous) =>
        data.error
          ? { ...previous, error: data.error, pending: false }
          : {
              ...data,
              probability: data.probabilities[0],
              baseline: data.probabilities[1],
              pending: false,
            },
      );
    };
    w.onerror = () =>
      setState((previous) => ({
        ...previous,
        pending: false,
        error: "The prediction worker stopped. Please retry.",
      }));
    return () => {
      w.terminate();
      worker.current = null;
    };
  }, [attempt]);
  useEffect(() => {
    const id = ++sequence.current;
    setState((previous) => ({ ...previous, pending: true, error: undefined }));
    const timer = setTimeout(
      () =>
        worker.current?.postMessage({
          id,
          graphs: baseline ? [graph, baseline] : [graph],
          base: new URL("model/", document.baseURI).href,
        }),
      180,
    );
    return () => clearTimeout(timer);
  }, [graph, baseline, attempt]);
  return { ...state, retry: () => setAttempt((a) => a + 1) };
}
