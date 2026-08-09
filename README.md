# LLM Controllability Geometry

This repository studies which internal states a language model can reach while
its verified behavior remains fixed. Prompt optimization is one control
channel. Direct residual interventions, feedback control, and prompt plus
activation control provide the others.

The central measurements are:

* behavior preserving reachable set rank and displacement
* active tangent expansion with uncertainty driven C-MAP queries
* bracketed and right censored behavior boundaries for ordered controls
* directed prompt to activation and activation to prompt accessibility gaps
* a standardized representation-control gap
* principal angles between prompt and activation reachable subspaces
* target aligned and orthogonal movement as control budget grows
* minimum control cost for a requested internal setpoint
* generation time setpoint error and dose response
* local residual control rank over nested orthonormal bases
* transfer across held out sources and subject categories
* monitor consistency over behavior equivalent states
* causal route comparison through state, path, component, and head interventions

The planned collection and analysis path is executable. The repository does
not contain final experiment results, and the paper does not claim them.

## Setup

The project uses `uv` for dependency resolution and command execution.

```bash
uv sync --extra remote --group dev
uv run llm-controllability --help
uv run python -m pytest -q
```

Gemma Scope analysis has one additional dependency group:

```bash
uv sync --extra remote --extra scope --group dev
```

### Apple silicon and MPS

On an Apple silicon Mac, install the same locked environment and verify that
PyTorch can use Metal:

```bash
uv sync --extra remote --group dev
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run llm-controllability runtime-info \
  --device-map mps \
  --torch-dtype float16 \
  --attn-implementation eager
```

The command must report `"selected_device": "mps"` and
`"mps_available": true`. Runtime selection defaults to `auto`, which chooses
CUDA when available, then MPS, then CPU. Explicitly requesting an unavailable
backend fails instead of silently running on CPU.

Run a bounded end-to-end pilot on MPS with:

```bash
bash scripts/run_mps_model_controllability.sh \
  google/gemma-3-4b-it \
  gemma3_4b_it
```

The MPS launcher uses eager attention, float16 weights, one-example batches,
MPS monitor training, and CPU fallback for PyTorch operations that Metal does
not implement. Pass `float16 full` after the slug to run the publication-scale
protocol instead. MPS bfloat16 requires macOS 14 or newer; earlier releases
automatically use float16.

MPS cannot offload part of a checkpoint to CPU through
`device_map="auto"`. The complete model and experiment working set must fit in
unified memory. Start with Gemma 3 4B or Phi 4 Mini. The 8B and 12B studies
require substantially more unified memory, especially during prompt
optimization. On a 48 GB Mac, the launcher uses search batch 1 for Qwen 3 8B
and Gemma 3 12B; Qwen direction fitting uses batch 2 at 8B. See the official
[PyTorch MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html) and
[Transformers Apple silicon guide](https://huggingface.co/docs/transformers/perf_train_special)
for current platform requirements.

## Full experiment path

Build the local benchmark bundle:

```bash
uv run llm-controllability build-frontier-data \
  --out-dir data/frontier \
  --sources mmlu_pro math500 gpqa_diamond hle \
  --allow-gated \
  --max-items-per-source 500 \
  --train-fraction 0.60 \
  --validation-fraction 0.20 \
  --behavior-limit 300
```

Run the complete recommended matrix:

```bash
bash scripts/run_recommended_matrix.sh
```

The matrix is declared in `configs/recommended_matrix.json`; the launch scripts
read model membership, protocol, precision, attention implementation, and
enabled analyses from that file. After all five runs finish, the script
validates every required artifact and writes combined model tables and
revision records under
`runs/controllability/matrix`.

The primary matrix is not used to fit a scaling trend. Run the separate four
size Qwen 3 protocol after the primary matrix:

```bash
bash scripts/run_scaling_matrix.sh
```

This reduced protocol reruns Qwen 3 0.6B, 1.7B, 4B, and 8B with the same five
seeds, contexts, example counts, and analysis budgets. It writes its aggregate under
`runs/controllability/scaling_matrix`. It reports fitted trends, leave one
size out stability, and an evidence qualification. A trend is never labeled a
law from fit quality alone.

Inspect the resolved commands without starting a model run:

```bash
uv run llm-controllability run-study-matrix \
  --analysis controllability \
  --dry-run
```

Set `MODEL_REVISION` to a Hugging Face commit when a study must be pinned
before download. If it is unset, direction fitting records the resolved commit
and carries it into the generated search and study specifications.

The five full studies are:

* Gemma 3 4B IT and 12B IT
* Phi 4 Mini Instruct
* Qwen 3 4B and 8B

Qwen 3 8B is the deep causal anchor. Gemma 3 12B is the single larger
validation point. The 0.6B, 1.7B, 4B, and 8B Qwen sequence is the only family
used for the scaling fit. No checkpoint exceeds 12B.

Run the token pooling comparison on all five primary checkpoints:

```bash
bash scripts/run_recommended_token_monitors.sh
```

Run the Gemma Scope 2 feature analysis on the two Gemma 3 checkpoints:

```bash
bash scripts/run_gemma_scope_matrix.sh
```

The scripts use five prompt search and C-MAP seeds, five monitor fits, and five
Jacobian control bases. Prompt controls are selected evenly across search seeds
and retain a provenance sidecar. Full layer direction fitting, hard prompt and
output semantic gates, task and quality gates, transfer analysis, deterministic
causal pair selection, held out monitor evaluation, and fixed figure generation
remain enabled. These are experiment commands, not smoke tests.
[docs/experiments.md](docs/experiments.md) gives the model matrix, artifact
checks, and reporting rules.

## Package layout

`src/llm_controllability/interventions` contains optimized suffixes, per example
prompt rewrites, fixed activation, directional ablation, PID, and hybrid
controls.

`src/llm_controllability/constraints` contains the hard task, semantic, output
quality, and control budget gates.

`src/llm_controllability/reachability` generates controlled outputs, recaptures the
resulting token sequences, actively expands unexplored local tangent spaces,
and computes effective rank, Jacobian rank, principal angles, curvature,
connectivity, behavior maps, and sampled boundary sharpness.

`src/llm_controllability/causal` contains activation caching, counterfactual and
cross prompt state replacement, receiver blocking, component ablation, and
head ablation.

`src/llm_controllability/monitors` contains last token, mean, max, learned attention,
multi layer, nonlinear, permuted label, and Mahalanobis OOD monitors.

`src/llm_controllability/evaluation` contains minimum control cost, dose response,
transfer, causal, invariance, corrected paired tests, and deterministic figure
generation.

`src/llm_controllability/models` selects the correct Hugging Face loader,
discovers nested text transformers, and applies each checkpoint's native prompt
template. Qwen thinking is disabled so hidden reasoning traces do not change
output length or control cost across the scaling series. The runtime resolver
supports CUDA, Apple MPS, and CPU and records requested and resolved settings
in each study manifest.

`src/llm_controllability/features` maps accepted Gemma 3 residual states into
Gemma Scope 2 sparse features, repeats the channel geometry analysis, and
evaluates a held out SAE feature monitor.

`src/llm_controllability/optimization`, `data`, `targets`, and `reporting`
separate prompt search, benchmark construction, target definitions, and
artifact rendering. Small deterministic assets live under `tests/fixtures`;
reported experiments use generated files under ignored `data/`.

EPO and GCG can optimize suffixes over sampled training contexts. Random search
and natural text scan remain available as prompt channel controls. The matrix
searches both decrease and increase objectives and retains eight controls from
each Pareto set without allowing one lucky seed to supply the entire set.

## State archives

`collect-reachable` writes:

* `states.npz`, with pooled states and optional packed token states
* `samples.jsonl`, with prompts, outputs, intervention settings, costs, tags,
  metrics, and every behavior gate verdict
* `geometry.csv`, with channel specific rank, displacement, preservation, and
  prompt to activation overlap
* `target_geometry.csv`, `budget_growth.csv`, `layer_propagation.csv`, and
  `principal_angles.csv`, with the remaining declared geometry measurements
* `trajectory_geometry.csv`, with ordered control path curvature
* `split_half_stability.csv` and `split_half_stability_summary.csv`, with the
  within channel reference for subspace and rank stability
* `controllability_boundaries.csv` and
  `controllability_boundary_summary.csv`, with finite sample boundary brackets,
  censoring status, binding constraints, and bootstrap intervals
* `directed_accessibility.csv` and `directed_accessibility_summary.csv`, with
  equal count subsampled nearest set gaps in both channel directions
* `detection_control_gap.csv` and `representation_control_gap.csv`, with
  natural projection separation and behavior preserving control margin in
  common units
* `controllability_atlas.csv`, with control authority stratified by concept,
  task category, source, layer, and channel
* `boundary_survival.csv` and `phase_transition_candidates.csv`, with
  bootstrap preservation curves and conservative sharp boundary qualification
* `cmap/`, `cmap_directions.npz`, `cmap_queries.csv`,
  `cmap_directions.csv`, and `cmap_summary.csv`, with active validation split
  discovery and held out test evaluation
* `manifest.json`, with run counts, exact revisions, software, device, and
  artifact names

Only states that pass every configured gate enter the reachable set geometry.
Failed interventions remain in the archive so that preservation rates and
failure modes are auditable.

## Local data

Model weights, benchmark rows, generated states, result tables, paper PDFs, and
posters are excluded from version control. Do not commit or redistribute gated
benchmark rows. The builder code and source manifest are the reproducible part
of the data pipeline.
