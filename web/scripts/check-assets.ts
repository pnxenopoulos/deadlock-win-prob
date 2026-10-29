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
const file = Bun.file(new URL("model/manifest.json", directory));
if (!(await file.exists()))
  throw new Error("Run bun run model before building the scenario website.");
const manifest = await file.json();
validateManifest(manifest);
validateEncoder(
  await Bun.file(new URL("model/encoder.json", directory)).json(),
);
for (const name of ["encoder.json", "model.onnx"]) {
  const data = await Bun.file(
    new URL("model/" + name, directory),
  ).arrayBuffer();
  if (
    createHash("sha256").update(new Uint8Array(data)).digest("hex") !==
    manifest.sha256[name]
  )
    throw new Error(`Model checksum mismatch: ${name}`);
}
console.log(`Verified game assets and model ${manifest.model_version}`);
