# LLM Controllability Geometry

This repository studies which internal states a language model can reach while
its verified behavior remains fixed. Prompt optimization is one control
channel. Direct residual interventions, feedback control, and prompt plus
activation control provide the others.

The central measurements are:

* behavior preserving reachable set rank and displacement
* principal angles between prompt and activation reachable subspaces
* target aligned and orthogonal movement as control budget grows
* minimum control cost for a requested internal setpoint
* generation time setpoint error and dose response
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
enabled analyses from that file. After all nine runs finish, the script
validates every required artifact and writes combined model tables, matched
post training contrasts, and revision records under
`runs/controllability/matrix`.

Inspect the resolved commands without starting a model run:

```bash
uv run llm-controllability run-study-matrix \
  --analysis controllability \
  --dry-run
```

Set `MODEL_REVISION` to a Hugging Face commit when a study must be pinned
before download. If it is unset, direction fitting records the resolved commit
and carries it into the generated search and study specifications.

This launches five full studies:

* Gemma 3 4B IT and 12B IT
* Gemma 4 12B IT
* Phi 4 Mini Instruct and Phi 4

It then launches four matched comparison checkpoints: Gemma 3 4B PT,
Gemma 3 12B PT, Gemma 4 12B PT, and Phi 4 Reasoning. These runs retain the
same search budget, seeds, behavior gates, state collection, and downstream
analyses. They omit only the random optimizer baselines that do not inform the
post training comparisons.

Run the token pooling comparison on all five primary checkpoints:

```bash
bash scripts/run_recommended_token_monitors.sh
```

Run the Gemma Scope 2 feature analysis on the four Gemma 3 matched
checkpoints:

```bash
bash scripts/run_gemma_scope_matrix.sh
```

The scripts use five search seeds, full layer direction fitting, hard prompt
and output semantic gates, task and quality gates, prompt and activation
sweeps, transfer analysis, causal reruns, held out monitor evaluation, and
fixed figure generation. They are experiment commands, not smoke tests.
[docs/experiments.md](docs/experiments.md) gives the model matrix, artifact
checks, and reporting rules.

## Package layout

`src/llm_controllability/interventions` contains optimized suffixes, per example
prompt rewrites, fixed activation, directional ablation, PID, and hybrid
controls.

`src/llm_controllability/constraints` contains the hard task, semantic, output
quality, and control budget gates.

`src/llm_controllability/reachability` generates controlled outputs, recaptures the
resulting token sequences, and computes effective rank, Jacobian rank,
principal angles, curvature, and connectivity.

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
template. Gemma 4 uses its multimodal loader in text only mode. Thinking is
disabled for Gemma 4 so its control budgets are comparable.
Phi 4 Reasoning keeps its native reasoning format because reasoning post
training is the variable in that matched comparison.

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
each Pareto set.

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
