export interface ModelOption {
  id: string;
  label: string;
  base: string;
  version?: string;
}

export const bundledGraph = (): ModelOption => ({
  id: "graph",
  label: "Graph transformer",
  base: new URL("model/", document.baseURI).href,
});

export async function graphModels(signal: AbortSignal): Promise<ModelOption[]> {
  const model = bundledGraph();
  const response = await fetch(new URL("models.json", model.base), { signal, cache: "no-cache" });
  if (!response.ok) return [model];
  const releases = (await response.json()) as { version: string; path: string }[];
  const models = releases
    .filter(
      ({ version, path }) =>
        typeof version === "string" && ["", "previous/"].includes(path),
    )
    .map(({ version, path }): ModelOption => ({
          id: path ? "graph-previous" : model.id,
      label: "Graph · " + version,
      base: new URL(path, model.base).href,
      version,
    }));
  return models.some(({ id }) => id === model.id) ? models : [model];
}
