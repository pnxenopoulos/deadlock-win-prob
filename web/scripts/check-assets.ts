import { createHash } from "node:crypto";
import { validateEncoder, validateManifest } from "../src/ml/features";
import catalog from "../src/data/catalog.json";
const directory = new URL("../public/", import.meta.url);
for (const path of [
  "minimap/minimap_midtown_mid_psd.webp",
  "minimap/minimap_midtown_mid_tunnels_psd.webp",
  ...catalog.heroes.map((h) => h.icon),
  ...catalog.items.map((i) => i.icon),
]) {
  if (!(await Bun.file(new URL(path.replace(/^\//, ""), directory)).exists()))
    throw new Error(`Missing game asset: ${path}`);
}
const index = Bun.file(new URL("model/models.json", directory));
const releases: { version?: string; path: string }[] = await index.exists()
  ? await index.json()
  : [{ path: "" }];
if (!releases.some(({ path }) => path === ""))
  throw new Error("The model index has no default graph model.");
for (const { path, version } of releases) {
  if (!["", "previous/"].includes(path))
    throw new Error("Invalid model path in the model index.");
  const base = new URL("model/" + path, directory);
  const file = Bun.file(new URL("manifest.json", base));
  if (!(await file.exists()))
    throw new Error("Run bun run model before building the scenario website.");
  const manifest = await file.json();
  validateManifest(manifest);
  if (version && version !== manifest.model_version)
    throw new Error("Model does not match the version in the model index.");
  validateEncoder(await Bun.file(new URL("encoder.json", base)).json());
  for (const name of ["encoder.json", "model.onnx"]) {
    const bytes = await Bun.file(new URL(name, base)).arrayBuffer();
    if (createHash("sha256").update(new Uint8Array(bytes)).digest("hex") !== manifest.sha256[name])
      throw new Error("Model checksum mismatch: " + name);
  }
  console.log("Verified game assets and model " + manifest.model_version);
}
