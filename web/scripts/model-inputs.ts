import { encodeGraphs, validateEncoder } from "../src/ml/features";
import { clone, presets, setHealth } from "../src/scenario";
import type { Encoder } from "../src/ml/types";
const encoder = (await Bun.file(
  new URL("../public/model/encoder.json", import.meta.url),
).json()) as Encoder;
validateEncoder(encoder);
const baseline = clone(presets[2].graph),
  edited = clone(baseline);
edited.players[0] = setHealth(edited.players[0], 0);
const inputs = encodeGraphs([baseline, clone(baseline), edited], encoder);
await Bun.write(
  Bun.argv[2],
  JSON.stringify(inputs, (_, value) =>
    typeof value === "bigint"
      ? Number(value)
      : ArrayBuffer.isView(value)
        ? Array.from(value as unknown as ArrayLike<number>)
        : value,
  ),
);
