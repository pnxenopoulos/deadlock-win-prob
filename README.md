# deadlock-win-prob
Win probability in Deadlock

Manifest requests and replay downloads identify this project
with `User-Agent: deadlock-win-prob (+https://github.com/pnxenopoulos/deadlock-win-prob)`.

The package is organized by responsibility:

```text
src/dwp/
├── data/                 # Data sources, replay preparation, and match splits
│   ├── deadlock_api.py
│   ├── replays.py
│   ├── datasets.py
│   ├── planning.py
│   ├── preparation.py
│   └── _http.py
├── graphs/               # Replay snapshots and raw graph storage
│   ├── snapshots.py
│   └── parquet.py
├── ml/                   # Feature encoding, modeling, and evaluation
│   ├── features.py
│   ├── batching.py
│   ├── model.py
│   ├── checkpoints.py
│   ├── export.py
│   ├── training.py
│   ├── metrics.py
│   ├── reporting.py
│   └── plots.py
└── cli/                  # Console commands
    ├── prepare.py
    └── train.py
```

Tests mirror these packages under `tests/`. Imports use these paths, for example
`from dwp.ml.checkpoints import load_model`. The `dwp-train`, `dwp-prepare`, and
`deadlock-win-prob` console commands are unchanged; module execution uses
`python -m dwp.cli.train` or `python -m dwp.cli.prepare`.

`dwp.data.deadlock_api` loads data from deadlock-api:

- `load_latest_match_salts()` loads the current match salts Parquet snapshot.
- `load_recent_matches(days=7)` returns unique `match_id`, `winning_team` pairs for
  TeamWin outcomes, defaulting to Ranked and Normal. Mode filters accept enum names.
  The day window uses file build times in UTC; files may contain older matches.

`dwp.data.replays` constructs replay URLs, selects matches, and downloads replays:

```python
from dwp.data.deadlock_api import load_latest_match_salts, load_recent_matches
from dwp.data.replays import add_replay_urls, select_replays

matches = load_recent_matches(days=7)
replays = add_replay_urls(load_latest_match_salts())
selected = select_replays(
    replays,
    match_ids=matches["match_id"].to_list(),
    sample_size=100,
    seed=42,
)
```

`selected` contains only `match_id` and `replay_url`, with one known URL per match.
Omit `sample_size` to select every matching replay. Sampling uses at most the number
of available matches, without replacement; `seed` makes the sample reproducible.

Use `dwp.data.replays.download_replay(replay_url, session=None)` to download and decompress a replay into bytes. Pass a `requests.Session` to reuse connections; the default uses `requests.get`.

Missing cluster IDs or replay salts produce null URLs. Source winner labels are
preserved, including occasional Spectator values on TeamWin rows.

Parse a replay URL with [Boon](https://boon.readthedocs.io/en/latest/api.html) using
an automatically cleaned temporary `.dem` file:

```python
from dwp.data.replays import load_replay

demo = load_replay(selected["replay_url"][0])
demo.load("kills", "damage")
kills = demo.kills  # Polars DataFrame.
```

`load_replay(url, session=None, preload=False)` returns a `boon.Demo`. Pass a
`requests.Session` to reuse connections or `preload=True` to load combat datasets
immediately. Boon retains the file's memory mapping for later dataset loads.

`dwp.graphs.snapshots` samples a Boon Demo into typed `GraphSnapshot` objects:

```python
from dwp.graphs.snapshots import iter_graphs

graphs = iter_graphs(demo, every=128, start_tick=1000, end_tick=20000)
graph = next(graphs)
players = graph.players
nodes = graph.nodes
edges = graph.edges
```

Each graph retains separate Polars feature tables:

| Table | Features |
| --- | --- |
| `global_info` | Pause-adjusted recording time, observed mid-boss alive state, recorded next-mid-boss timer. |
| `players` | Account/hero/team, XYZ, health/max health/barrier, alive/respawn state, level, souls/economy, recorded cooldown fields, and current upgrade tokens. |
| `objectives` | Type, lane, team, health/max health, XYZ, patron phase, destruction flag, and entity lifetime. |
| `ziplines` | Full node handle, lane, XYZ, control team, connections, capturability, and recorded cumulative distance. |

`nodes` provides the shared node index. `edges` materializes every directed pair
except self-loops, with source/target types and teams, team-relation labels, and
3D distance. Global-node distances are null. Complete edge tables grow as
N * (N - 1), so consume the iterator and edge tables one graph at a time.
Zipline connections remain available as features alongside the complete edges.

Sampling follows Boon's tick stride, anchored to the first recorded tick; windows
are inclusive. Paused ticks and ticks without known pause state are skipped
before graph construction and optional stat enrichment; `is_paused` is not a
model input. Objectives use backward as-of observations, retaining destroyed
objectives and distinguishing reused entity slots by serial. Zipline presence and
control use Boon's causal snapshots. Final winners and future event state are
excluded from features. Each player carries `team_num`: 2 is Hidden King and
3 is Archmother; missing teams stay null. Teams enter the model through a separate
categorical embedding and its team-aware attention/readout, rather than the numeric
feature list. Unknown or ambiguous player accounts receive frame-local node IDs rather than false
cross-tick identities.

Pass an explicit matching boon-data `data_version` to classify `item_ids` and add
per-player `baseline_stats`; `baseline_stats=()` selects item classification only.
These queries can download that catalog. Without a version, both columns are null
and the original `upgrade_ids` remain available: these tokens mix items and
abilities. Calculated stats retain their units, status, and diagnostics; unresolved
values stay null. Boon's baseline includes owned passive items and permanent
bonuses, rather than only untouched hero stats.

`respawn_time` and `next_midboss` preserve Boon's raw recorded timers, without
assuming they are countdowns. `elapsed_seconds` uses only the tick rate and pauses
before the current tick, without dividing by final match duration. Boon's
`match_time_seconds` reconstructs a pregame offset using game-over information,
so it is excluded from graph/model features and sampling decisions. It is used
only for post-inference display labels in the prediction table, where unavailable
match clocks stay null. An unobserved mid-boss state stays null.

Useful next features for win probability are ordinary ability/item cooldowns,
trooper and lane pressure, urn/Rift state, and team-specific visibility. Player
level/economy/barriers and objective phase/max health are already retained. These
graphs describe the recorded replay state; a team-view model would also need
explicit observation masks.

The prediction code is split into small modules:

- `dwp.ml.features`: `GraphEncoder.fit(training_graphs)` learns numerical scaling
  and categorical vocabularies from training graphs only. `transform(graph)`
  returns `GraphTensors`, with missing-value masks and no predictive identifiers.
- `dwp.ml.batching`: `GraphBatch.from_graphs(graphs)` pads each node type and
  upgrade-token list into a tensor batch. `.to(device)` transfers the collated
  batch; its validity mask excludes padding from attention and pooling.
- `dwp.ml.model`: `HeterogeneousGraphTransformer(encoder)` defaults to three layers,
  128 hidden dimensions, four attention heads, and dropout 0.1. It uses separate
  node-type encoders and Q/K/V projections, pooled upgrade-token embeddings,
  team/type/geometry/connectivity attention biases, and separate readout buckets
  for each node type and team. `model(graphs)` returns one **logit** per graph;
  `model(graphs).sigmoid()` returns P(Archmother wins).
- `dwp.data.datasets`: `split_matches(rows)` partitions whole match IDs into roughly
  80% training, 10% validation, and 10% test. All rows and columns are retained,
  including repeated snapshots. Each split has at least one distinct match.
  Set `seed` for reproducible shuffling, or `order_by="start_time"` to split
  chronologically using an actual match timestamp, constant within each match.
- `dwp.ml.metrics`: `probability_metrics(labels, probabilities, bins=10)` returns
  log loss, ECE, Brier score, and a calibration-bin table.
  `plot_calibration(result)` returns a Matplotlib figure with a reliability
  diagram and the prediction distribution; export it with `figure.savefig(...)`.

Create binary labels with `winner_label(demo.winning_team_num)` for known team
wins: native Boon team 3 (Archmother) becomes 1, team 2 (Hidden King) becomes 0.
Unknown winners and other outcomes must be excluded. This helper deliberately
rejects other team numbers; deadlock-api's Team0/Team1 strings require their own
explicit mapping. Give every pre-outcome snapshot the match's label, keeping the
label outside its graph features. Exclude snapshots at/after the victory event.

Run the full data-to-training pipeline with the installed command (also available
as `python -m dwp.cli.train` or `deadlock-win-prob`):

```bash
uv run --locked dwp-train --days 1 --matches 50 --epochs 20 \
  --every 4096 --max-snapshots 16 --batch-size 4 --output runs/example
```

`--match-modes` defaults to `Ranked`, and `--game-modes` to `Normal`; both accept
multiple enum names. Other options include `--seed`, `--learning-rate`,
`--hidden-dim`, `--num-heads`, `--num-layers`, `--dropout`, `--bins`,
`--train-fraction`, `--validation-fraction`, `--threads`, `--device` (auto/cpu/cuda),
and an optional matching catalog `--data-version`. Use `--help` for all defaults.
The output directory must be new; its default is a timestamped directory in
`runs/`. Training outputs are ignored by Git.

The script uses `dwp.data.preparation` to load recent candidates and salts, then
construct URLs only for those candidates. It collects at most `--candidate-limit`
distinct candidates (default `max(100, 20 * matches)`), visiting newer Parquet
files first, and shuffles the eligible replay URLs. This bounds data collection;
it is not uniform random sampling over the entire data lake. As in the data API,
`--days` refers to file build age, not match start time. Increase the candidate
limit when too few usable replays are found.

Replay downloads reuse a requests session and detect bzip2 or Zstandard by file
contents, including current Zstandard replays whose URL still ends in `.bz2`.
Expired downloads, parse errors, mismatched match IDs, missing game-over events,
and unsupported winners are logged and replaced with other candidates. It uses
up to `--matches` successful replays and requires at least three. Boon's native
winner supplies the label, and snapshots stop strictly before `game_over_tick`.
Each match retains a reproducible random subset of at most `--max-snapshots`.

`dwp.ml.training` splits whole matches, fits feature encoding on the training split,
and samples one snapshot per training match each epoch. Validation uses all
retained validation snapshots with equal total weight per match. The checkpoint
with the lowest validation log loss is restored before the final test evaluation.
The test set never participates in fitting or checkpoint selection. Small runs
are useful pipeline checks; use substantially more matches for meaningful scores.

Each completed run saves:

- `model.pt`: weights, architecture settings, feature schema, normalization and
  vocabularies; restored with `weights_only=True` by `dwp.ml.checkpoints.load_model`.
- `metrics.json`: validation/test log loss, ECE, Brier score, sample counts, and
  the selected epoch; `summary.md` provides a readable table for CI.
- `validation_calibration.png` and `test_calibration.png`, plus calibration-bin
  tables and per-snapshot predictions as CSV files for each split.
- `matches.csv`: match IDs, labels, replay URLs, builds, game-over cutoffs, split
  membership, and snapshot counts. `run.json` records settings and package versions;
  `history.json` records training loss and validation scores per epoch.

Checkpoints require the same feature schema as the code; retrain after changing
the feature list. Regenerate parsed graph shards when snapshot-sampling rules change.

Load the saved model and fitted encoder for inference:

```python
import torch
from pathlib import Path
from dwp.ml.checkpoints import load_model

model, encoder = load_model(Path("runs/example/model.pt"), device="cpu")
with torch.inference_mode():
    probability = model(encoder.transform(graph)).sigmoid().item()
```

For parallel preparation, `dwp-prepare` separates source discovery from parsing:

```bash
uv run --locked dwp-prepare plan --days 1 --matches 20 --workers 4 \
  --max-parallel 2 --every 4096 --max-snapshots 16 --output runs/plan.json
uv run --locked dwp-prepare shard --plan runs/plan.json --shard-id 0 \
  --output runs/prepared/shard-0
# Run shard 1, 2, and 3 independently with their own output directories.
uv run --locked dwp-train --prepared-data runs/prepared --plan runs/plan.json \
  --epochs 10 --output runs/model
```

The plan is a fingerprinted manifest of disjoint candidate URL pools, per-worker
match quotas, sampling settings, and source parameters. `--matches` is a global
maximum, not a per-worker count. The planner queries recent matches and salts
once, then assigns both primary and replacement candidates. The actual worker
count is at most the requested workers, target matches, and eligible URL count;
workers may produce zero, one, or more usable matches. At least three usable
matches are required across the combined dataset.

Each worker saves `matches.parquet` and four raw feature tables:
`global_info.parquet`, `players.parquet`, `objectives.parquet`, and
`ziplines.parquet`. Nested inventories/baselines, nulls, node order, and graph
identities survive the Parquet handoff. `dataset.json` is written last as the
completion marker. Workers never fit feature encoders or train a model.
`dwp.graphs.parquet.load_graph_shards` combines all planned workers, rejecting missing
shards, mixed plans, duplicate match IDs, unassigned matches, and post-game states.
The loader converts one raw shard table at a time to temporary uncompressed IPC
files, then reads only requested snapshot row ranges. Each match's `graphs` is a
lazy sequence: indexing, iteration, and `random.choice` work without retaining
all snapshot DataFrames. Keep the returned matches alive while training; their
temporary files are removed when released. This needs extra local temporary disk
space for the uncompressed tables, but does not cache encoded features.
The final trainer reads these files without downloading or reparsing replays,
splits whole matches, and fits its encoder on training matches only. It saves
the same best-validation checkpoint, test metrics, and plots as the local pipeline.

`dwp.ml.plots` can plot a demo's win probabilities every 10 seconds:

```python
from pathlib import Path
from boon import Demo
from dwp.ml.checkpoints import load_model
from dwp.ml.plots import plot_win_probability

model, encoder = load_model(Path("runs/example/model.pt"))
demo = Demo("match.dem")
figure = plot_win_probability(demo, model, encoder)
figure.savefig("win_probability.png", dpi=150)
```

The figure shows Archmother and Hidden King probabilities against match time in
minutes, with recording time as a fallback for partial/older demos. Sampling uses
Boon's pause-adjusted recording clock and explicit active ticks, starting at the
first known active recorded tick. A target coinciding with a pause uses the first
resumed tick. It stops before game over, or at recording end if no game-over event
is present. Empty player states are skipped, and no winner endpoint is added.
Match time is reconstructed after prediction solely for plot/CSV labels; changing
that future-derived offset cannot change sampled ticks or model predictions.
Change the default with `interval_seconds=5`, for example. Use the same optional
`data_version` as training when baseline/catalog features were enabled. Inference
reuses the fitted encoder, runs without gradients, and restores the model's mode.

`predict_win_probabilities(demo, model, encoder)` from the same module returns a
Polars table with `tick`, `elapsed_seconds`, nullable `match_time_seconds`,
`archmother_win_probability`, and `hidden_king_win_probability` for CSV export or
custom plots. Both functions accept the same interval and catalog options.

The manual GitHub Actions workflow is `.github/workflows/train.yml`. Once it is
on the default branch, select **Actions → Train win probability model → Run
workflow**. In addition to days, matches, epochs, sampling, batch size, seed, and
mode inputs, it accepts **workers** (default 4) and **max_parallel** (default 4).
The workflow runs a planner, then a preparation matrix capped by `max_parallel`,
then one training/evaluation job after every preparation job succeeds. It uses
CPU runners. Data-source failures are logged and replaced within each worker's
assigned pool; a worker may finish below its quota, and the actual counts are
recorded in its artifact and the final training manifest.

Preparation jobs upload uniquely named Parquet artifacts. Artifact names include
the plan fingerprint, so the final job cannot accidentally mix plans on reruns;
retries can replace only their own shard artifacts. Shards are downloaded into
separate directories. Missing/failed jobs prevent final training, and the loader
also verifies plan completeness. The workflow retains the plan, parsed graphs,
worker logs, and final model/report artifacts for 14 days, and reports held-out
log loss and ECE in the final job summary. Training is only triggered manually.
No repository secrets are needed for the public data source.

The training job logs shard indexing, feature fitting, and training/evaluation
progress, plus memory, CPU, and disk usage every minute. Resource logs and peak
process memory from `/usr/bin/time` are included with the final artifact when
upload can run. Exit code 143 means SIGTERM; a runner-shutdown message does not
by itself establish an out-of-memory kill. Keep `batch_size` at 1 or 2 for these
CPU runners before increasing it: dense attention memory scales with batch size
and the square of the largest padded graph. Increasing preparation workers does
not reduce memory used by a training batch.

Reuse the fitted encoder unchanged for validation, test, and inference, and save
it alongside the model checkpoint. Numeric features use signed `log1p` followed
by training-set standardization. Missing values receive zero with an observed
mask; unseen categories use an unknown token. Current `upgrade_ids` pool items
and abilities without requiring a catalog; optional classified item counts and
the six baseline stats are included when supplied by the representations module.
Positions are also supplied as relative coordinates and distances, scaled by
10,000 game units. Missing geometry is masked, and zipline connectivity preserves
known-empty versus unknown connection lists. Raw recorded timers stay raw.

Attention is dense within each graph and excludes self-edges; residual connections
retain each node's own state. Graphs share batched projections, attention, and
readout, with each graph in its own batch entry. Node types and upgrade lists are
padded independently; masks exclude padded nodes from attention and pooling.
Cost is proportional to batch size times the squared padded node count, so start
with small batches. Existing model checkpoints retain the same weights and schema.

`--batch-size` controls actual tensor batching for training, validation, and test
evaluation. Training still selects one random snapshot per match per epoch;
evaluation still scores every retained snapshot with equal total weight per match.
Features are encoded on demand for each bounded batch, without a feature cache.
The same setting is exposed as `batch_size` in the manual GitHub Actions workflow.
Batching throughput depends on the device: benchmark the intended runner before
increasing batch size. Larger batches can be slower on small CPU runners; the
local CUDA benchmark benefits more. `--batch-size 1` remains supported.

`model(graph)` and `model([graph_a, graph_b])` collate automatically. To collate on
CPU before transferring a whole batch to a GPU:

```python
from itertools import islice
from dwp.ml.batching import GraphBatch

batch = GraphBatch.from_graphs(
    [encoder.transform(g) for g in islice(graphs, 4)]
).to(device)
logits = model.to(device)(batch)
```

ECE uses equal-width bins and compares **mean P(Archmother wins)** with the
**observed Archmother win rate**, weighted by bin mass. Bins are left-closed and
right-open, except the final bin includes 1. Empty/zero-weight bins have null
means and contribute nothing to ECE. Log loss clips endpoints only for numerical
stability. Scores default to equal snapshot weighting; supply weights as above
to prevent longer matches from dominating. ECE depends on binning and sample
size; also examine reliability plots and scores by game time. These are
calibration diagnostics; they do not fit a probability calibration transform.


## ONNX export and model releases

Export an existing best-validation checkpoint without retraining:

```bash
uv run --locked --extra export dwp-export runs/example/model.pt \
  --output runs/example/onnx --model-version my-model-v1
```

ONNX tooling is an optional `export` extra; normal inference, training, and
preparation do not require it. `dwp.ml.export.export_onnx(checkpoint, output,
model_version=...)` provides the same operation in Python. The output directory
must be new. Export loads the checkpoint on CPU, checks the ONNX graph, and
compares ONNX Runtime probabilities against PyTorch across different batch,
node, and inventory sizes, including masked/global-only cases. A failure prevents
release publication. Export always uses the saved checkpoint selected by
validation loss; it does not select weights using the test set.

The output contains:

- `model.onnx`: a single-file ONNX model, opset 18, with embedded weights.
- `encoder.json`: ordered numeric/categorical features, fitted means/scales,
  vocabularies, team mappings, missing-value rules, derived features, geometry,
  inventory, padding, and time-sampling conventions.
- `manifest.json`: model version, input names/dtypes/shapes, dimension rules,
  output labels, model settings, optional catalog version, software versions,
  parity results, and SHA-256 digests.
- `model.pt`: the original PyTorch checkpoint, byte for byte.
- `parity.json`, `SHA256SUMS`, and available `run.json`, `metrics.json`,
  `summary.md`: validation and training provenance.
- `win-probability-model.zip`: the complete, flat bundle of those files.

The ONNX model returns one `float32` P(Archmother wins) per graph. Hidden King's
probability is its complement. Inputs are encoded tensors; raw replay parsing and
encoder preprocessing remain the caller's responsibility. Float inputs are
`float32`, category/team/token inputs are `int64`, and masks/connections are
boolean. JavaScript clients need `BigInt64Array` for `int64` and `Uint8Array` for
boolean tensors in ONNX Runtime Web.

Numeric node tensors concatenate standardized features followed by observed
masks, in the exact order in `encoder.json`. Nodes are concatenated as global,
player, objective, zipline. All node-axis tensors use
`N = 1 + P + O + Z`; connection tensors have shape `[B, N, N]`. Every node type
has at least one slot, even when absent, and every player has at least one token
slot. Zero-filled placeholder nodes must have `valid=false`. Padding positions
are not real entities, and changing graph sizes must not truncate entities.
`dwp.ml.export.onnx_inputs(encoded_graphs)` produces this contract from Python
`GraphTensors`, including padding empty types. Model-internal relations,
attention, pooling, and sigmoid are included in the ONNX graph.

For a browser integration, serve `model.onnx`, `encoder.json`, and `manifest.json`
together from your site's static assets, pin a model version, and verify the
hashes. Export does not port Boon's feature extraction into the viewer. Match
the parser/catalog version and the full feature contract before inference.
WebGPU execution requires separate operator-coverage and performance validation.

Every successful manual training workflow exports the browser bundle and adds
it to the usual Actions artifact. A separate release job checks its checksums,
uploads the individual bundle files and ZIP, then publishes the GitHub release
`model-<run_id>-<run_attempt>` and marks it latest. Only that job receives
`contents: write`; preparation/training retain read-only repository permissions.
The release is created as a draft until asset upload succeeds. Retries can repair
a partial draft; already-published versions are left unchanged. Release assets
persist independently of the Actions artifacts' 14-day retention.

To run all export tests locally:

```bash
uv run --locked --extra export pytest
```


## Scenario website

`web/` is a static scenario editor in the style of deadlock.nyc. It runs the
released ONNX model locally in a Web Worker, with WebGPU and a CPU fallback.
No replay upload or server inference is needed. The map, hero portraits, and
item icons are reused from deadlock.nyc; game assets remain Valve's property.
See `web/public/NOTICE.txt` for provenance.

```bash
cd web
bun install --frozen-lockfile
bun run model          # download and checksum-verify the latest compatible release
bun run dev            # loopback-only development server
bun test
bun run build          # output: web/dist
```

The editor supports hero placement and recorded stats, inventory upgrade tokens,
objective health/destruction, zipline ownership on the real directed topology,
match time, Mid Boss state, JSON import/export, a local saved draft, and baseline
comparisons. Three identity-free replay snapshots supply starting presets. Item
edits preserve non-item upgrades and leave item-count/baseline fields missing,
matching the current no-catalog model; stats and economy are edited separately.
Hero swaps clear hero-specific ability tokens. Blank numeric fields stay missing.
The map/catalog template is a particular game version, not a universal patch map.

`.github/workflows/pages.yml` builds and deploys to GitHub Pages on website
changes or manual dispatch. Configure this repository's Pages source as
**GitHub Actions**. The model release workflow calls this same deployment after
publishing a validated model, passing its exact version. This explicit call is
needed because releases created with `GITHUB_TOKEN` do not trigger another
release-event workflow. Each site deployment bundles the model, manifest, and
encoder together from checksum-verified release assets; compatible models are
not downloaded directly from GitHub by visitors. Unsupported feature schemas
fail the build rather than silently using the wrong encoder.

Scenarios are conditional predictions, not combat simulations or causal
estimates. Unusual or inconsistent states can be outside the training distribution.
