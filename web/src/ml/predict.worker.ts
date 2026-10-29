import { encodeGraphs, validateEncoder, validateManifest } from "./features";
import { createRuntime, fallBackToCPU, type Runtime } from "./runtime";
import type { Encoder, Manifest, RawGraph } from "./types";
interface Request {
  id: number;
  graphs: RawGraph[];
  base: string;
}
let pending: Request | null = null,
  busy = false;
let loaded:
  | Promise<{ runtime: Runtime; encoder: Encoder; manifest: Manifest }>
  | undefined;
async function load(base: string) {
  const response = await fetch(new URL("manifest.json", base), {
    cache: "no-cache",
  });
  if (!response.ok) throw new Error("The model is unavailable. Please retry.");
  const manifest = (await response.json()) as Manifest;
  validateManifest(manifest);
  const verified = async (name: string) => {
    const r = await fetch(
      new URL(`${name}?v=${encodeURIComponent(manifest.model_version)}`, base),
    );
    if (!r.ok)
      throw new Error("The model could not be downloaded. Please retry.");
    const bytes = await r.arrayBuffer();
    const digest = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      (b) => b.toString(16).padStart(2, "0"),
    ).join("");
    if (digest !== manifest.sha256[name])
      throw new Error(
        "Model files changed during download. Please reload this page.",
      );
    return bytes;
  };
  const [bytes, metadata] = await Promise.all([
    verified("model.onnx"),
    verified("encoder.json"),
  ]);
  const encoder = JSON.parse(new TextDecoder().decode(metadata)) as Encoder;
  validateEncoder(encoder);
  return { runtime: await createRuntime(bytes), encoder, manifest };
}
async function drain() {
  if (busy) return;
  busy = true;
  while (pending) {
    const request = pending;
    pending = null;
    try {
      const { runtime, encoder, manifest } = await (loaded ??= load(
        request.base,
      ).catch((e) => {
        loaded = undefined;
        throw e;
      }));
      const encoded = encodeGraphs(request.graphs, encoder);
      const inputs = Object.fromEntries(
        Object.entries(encoded).map(([name, value]) => [
          name,
          new runtime.ort.Tensor(value.type, value.data, value.dims),
        ]),
      );
      let output;
      try {
        try {
          output = await runtime.session.run(inputs);
        } catch (error) {
          if (runtime.backend !== "webgpu") throw error;
          await fallBackToCPU(runtime);
          output = await runtime.session.run(inputs);
        }
        const probabilities = Array.from(
          output.archmother_win_probability.data as Float32Array,
        );
        if (
          probabilities.length !== request.graphs.length ||
          probabilities.some((p) => !Number.isFinite(p) || p < 0 || p > 1)
        )
          throw new Error("The model returned an invalid estimate.");
        self.postMessage({
          id: request.id,
          probabilities,
          modelVersion: manifest.model_version,
          heroIds: Object.keys(encoder.vocabularies.hero_id),
          upgradeIds: Object.keys(encoder.vocabularies.upgrade_ids),
        });
      } finally {
        for (const tensor of Object.values(inputs)) tensor.dispose();
        if (output)
          for (const tensor of Object.values(output)) tensor.dispose();
      }
    } catch (error) {
      self.postMessage({
        id: request.id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
  busy = false;
}
self.onmessage = (event: MessageEvent<Request>) => {
  pending = event.data;
  void drain();
};
