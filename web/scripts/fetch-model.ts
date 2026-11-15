import { validateEncoder, validateManifest } from "../src/ml/features";
/** Mirror verified public release assets for browser CORS compatibility. No demo data. */
import { mkdir, mkdtemp, rename, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
const directory = fileURLToPath(new URL("../public/model", import.meta.url));
const version = process.env.DWP_MODEL_VERSION || "latest";
if (
  version !== "latest" &&
  !/^model\/graph-win\/v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/.test(version) &&
  !/^model-[0-9]+-[0-9]+$/.test(version)
)
  throw new Error("Unsupported model release tag");
const api = "https://api.github.com/repos/pnxenopoulos/deadlock-win-prob/releases";
type Release = {
  tag_name: string;
  draft: boolean;
  prerelease: boolean;
  assets: { name: string; url: string; digest?: string }[];
};
const headers: Record<string, string> = {
  Accept: "application/vnd.github+json",
  "User-Agent": "deadlock-scenarios model-build",
};
if (process.env.GITHUB_TOKEN)
  headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
async function releaseData<T>(path: string): Promise<T> {
  const response = await fetch(api + path, { headers });
  if (!response.ok)
    throw new Error("Model release discovery failed (" + response.status + ").");
  return (await response.json()) as T;
}
const names = ["manifest.json", "encoder.json", "model.onnx"];
const isGraph = (release: Release) =>
  !release.draft && names.every((name) =>
    release.assets.some((asset) => asset.name === name),
  );
const [releases, pinned] = await Promise.all([
  releaseData<Release[]>("?per_page=30"),
  version === "latest"
    ? Promise.resolve(null)
    : releaseData<Release>("/tags/" + encodeURIComponent(version)),
]);
const available = releases.filter((release) => !release.prerelease && isGraph(release));
const primary = pinned ?? available[0];
if (!primary || !isGraph(primary))
  throw new Error("The selected release has no published graph model.");
const selected = [primary, ...available.filter((release) =>
  release.tag_name !== primary.tag_name,
)].slice(0, 2);

async function bundle(release: Release, target: string) {
  const bytes: Record<string, Uint8Array> = {};
  for (const name of names) {
    const asset = release.assets.find((a) => a.name === name);
    if (!asset || !asset.digest?.startsWith("sha256:"))
      throw new Error(`Missing model release asset or checksum: ${name}`);
    const response = await fetch(asset.url, {
      headers: { ...headers, Accept: "application/octet-stream" },
    });
    if (!response.ok)
      throw new Error(`Model download failed: ${name} (${response.status})`);
    const data = new Uint8Array(await response.arrayBuffer());
    if (
      "sha256:" + createHash("sha256").update(data).digest("hex") !==
      asset.digest
    )
      throw new Error(`Model release checksum mismatch: ${name}`);
    bytes[name] = data;
  }
  const manifest = JSON.parse(new TextDecoder().decode(bytes["manifest.json"]));
  if (
    manifest.format_version !== 1 ||
    manifest.model_version !== release.tag_name
  )
    throw new Error("Unsupported model release manifest.");
  validateManifest(manifest);
  validateEncoder(JSON.parse(new TextDecoder().decode(bytes["encoder.json"])));
  for (const name of ["model.onnx", "encoder.json"])
    if (
      createHash("sha256").update(bytes[name]).digest("hex") !==
      manifest.sha256[name]
    )
      throw new Error(`Mixed model assets: ${name}`);
  await mkdir(target, { recursive: true });
  for (const [name, data] of Object.entries(bytes))
    await Bun.write(target + "/" + name, data);
  console.log("Bundled graph model " + release.tag_name);
}

const temporary = await mkdtemp(directory.replace(/\/$/, "") + ".download-");
try {
  const models = [];
  for (const [index, release] of selected.entries()) {
    const path = index === 0 ? "" : "previous/";
    await bundle(release, temporary + "/" + path);
    models.push({ version: release.tag_name, path });
  }
  await Bun.write(temporary + "/models.json", JSON.stringify(models));
  await rm(directory, { recursive: true, force: true });
  await rename(temporary, directory);
} finally {
  await rm(temporary, { recursive: true, force: true });
}
