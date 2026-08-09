# Full controllability experiment runbook

This runbook produces the evidence required by the paper. The files in
`tests/fixtures/` are test assets and must not be used for reported results.

## 1. Environment and access

Install the locked environment:

```bash
uv sync --extra remote --group dev
```

Record the software and device state before the first run:

```bash
uv run python -VV
uv run llm-controllability runtime-info
git rev-parse HEAD
```

On Apple silicon, verify MPS explicitly before downloading a checkpoint:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run llm-controllability runtime-info \
  --device-map mps \
  --torch-dtype float16 \
  --attn-implementation eager
```

For a complete MPS experiment, use:

```bash
bash scripts/run_mps_model_controllability.sh \
  google/gemma-3-4b-it \
  gemma3_4b_it \
  float16 \
  full
```

The launcher sets `DEVICE_MAP=mps`, uses MPS for the learned attention monitor,
enables CPU fallback for unsupported Metal operators, and reduces search
batches to one. The full model and optimizer working set must fit in unified
memory because MPS cannot partially offload a checkpoint to CPU through the
automatic Hugging Face device map.

Omit the final `float16 full` arguments for a local pilot: one seed, four
optimization contexts, 50 GCG iterations, 128 study examples, two prompt
controls per direction, and 64 generated tokens. Pilot results validate the
pipeline but are not substitutes for the reported protocol.

Install the optional Gemma Scope dependencies before the feature analysis:

```bash
uv sync --extra remote --extra scope --group dev
```

Accept the Hugging Face terms for GPQA, Humanity's Last Exam, and the Gemma
checkpoints before using those sources or models. Authenticate with
`hf auth login`.

## 2. Build the benchmark bundle

```bash
uv run llm-controllability build-frontier-data \
  --out-dir data/frontier \
  --sources mmlu_pro math500 gpqa_diamond hle \
  --allow-gated \
  --max-items-per-source 500 \
  --train-fraction 0.60 \
  --validation-fraction 0.20 \
  --behavior-limit 300 \
  --seed 0
```

The builder writes source stratified train, validation, and test partitions.
Direction fitting uses the training contrast, and layer selection uses the
validation contrast. The test contrast remains untouched. Per example prompt
maps provide a natural paraphrase, concise style, capitalization, and matched
topic control without changing the question or answer. GPQA choices are
shuffled with a stable question hash and the correct letter is recomputed
after shuffling. HLE rows that require image or media inputs are excluded from
this text only study.

Required files:

```text
data/frontier/manifest.json
data/frontier/contrasts/eval_awareness_train.json
data/frontier/contrasts/eval_awareness_validation.json
data/frontier/contrasts/eval_awareness_test.json
data/frontier/behavior/controllability_train.jsonl
data/frontier/behavior/controllability_validation.jsonl
data/frontier/behavior/controllability_test.jsonl
data/frontier/behavior/controllability_all.jsonl
data/frontier/controls/prompt_rewrites.json
```

Do not commit `data/frontier`.

## 3. Model matrix

Five checkpoints receive the complete protocol. Qwen 3 8B is the deep causal
anchor, while Gemma 3 12B is the single larger validation point:

| Role | Hugging Face checkpoint | Slug |
| --- | --- | --- |
| Cross family check | `google/gemma-3-4b-it` | `gemma3_4b_it` |
| Larger validation | `google/gemma-3-12b-it` | `gemma3_12b_it` |
| Cross family check | `microsoft/Phi-4-mini-instruct` | `phi4_mini_instruct` |
| Qwen scale anchor | `Qwen/Qwen3-4B` | `qwen3_4b` |
| Deep causal anchor | `Qwen/Qwen3-8B` | `qwen3_8b` |

Run all five checkpoints in the declared order:

```bash
bash scripts/run_recommended_matrix.sh
```

The source of truth for membership, family labels, and protocol tier is
`configs/recommended_matrix.json`. It also declares precision,
attention implementation, and which downstream analyses apply to each model.
The script finishes with
`aggregate-study-matrix`, which rejects a missing or empty required artifact
before writing combined tables.

Resolve the commands without launching GPU work:

```bash
uv run llm-controllability run-study-matrix \
  --analysis controllability \
  --dry-run
```

To run one checkpoint, use:

```bash
bash scripts/run_model_controllability.sh \
  google/gemma-3-4b-it gemma3_4b_it bfloat16 eager full
```

Pin a checkpoint before download by setting `MODEL_REVISION` to its Hugging
Face commit. With no override, direction fitting records the resolved commit
in `direction_manifest.json`; the generated target and study specifications
reuse that commit.

The final argument is `full`, `matched`, or `scaling`. The primary matrix uses
`full`: five seeds, the complete optimizer comparison, all behavior gates,
reachable state collection, active C-MAP discovery, and every downstream
analysis. `matched` remains available for future controlled checkpoint pairs
but is not part of the declared matrix.

The cross family checkpoints test whether the main 8B findings are confined
to Qwen. They do not enter the size fit. The scaling hypothesis uses only four
instruction tuned Qwen 3 checkpoints:

| Size | Hugging Face checkpoint | Slug |
| --- | --- | --- |
| 0.6B | `Qwen/Qwen3-0.6B` | `qwen3_0_6b_scaling` |
| 1.7B | `Qwen/Qwen3-1.7B` | `qwen3_1_7b_scaling` |
| 4B | `Qwen/Qwen3-4B` | `qwen3_4b_scaling` |
| 8B | `Qwen/Qwen3-8B` | `qwen3_8b_scaling` |

```bash
bash scripts/run_scaling_matrix.sh
```

`configs/scaling_matrix.json` reruns all four sizes and writes the fit to
`runs/controllability/scaling_matrix`. The distinct scaling slugs prevent full
primary artifacts from entering the fit. Every size uses five optimizer seeds,
eight search contexts, 128 behavior examples, and a four direction C-MAP
budget. This matched budget is required to separate size from search effort.
The series is a targeted hypothesis test, not a replacement for the deep 8B
analysis.

Each script performs the following work without a smoke run substitution:

1. It fits the paired direction at every transformer layer.
2. It retains the strongest layer and runs EPO, GCG, equal budget random
   search, uniform random prompts, and natural text scanning in both projection
   directions.
3. It exports eight decrease and eight increase controls from seed-specific
   Pareto sets in round-robin seed order and writes source seed provenance.
4. It creates fixed addition, directional ablation, PID, orthogonal random,
   per example prompt rewrite, optimized prompt, and hybrid sweeps.
5. It generates one deterministic output, snapshots generation cost and
   controller tracking error, resets controller state, and recaptures the exact
   sequence. The reset prevents feedback accumulated during generation from
   leaking into a separate full sequence replay.
6. It admits a state only when task correctness, prompt semantics, output
   semantics, output quality, and the control budget all pass.
7. For each of five seeds, it actively proposes residual directions outside the
   observed tangent span, brackets their validation split behavior boundary,
   and evaluates accepted directions on held out test examples with C-MAP.
8. It computes setpoint reachability, censored behavior boundaries, directed
   channel accessibility, the detection control gap, dose response, budget
   growth, the controllability atlas, boundary sharpness, layer propagation,
   source and subject transfer, local Jacobian spectra, causal route tests,
   monitor invariance, and reachable state augmentation.
9. It renders the predeclared figures directly from the archived tables.

The prompt search averages every objective over 16 training contexts sampled
per seed. It uses sequence length 32, population 24, 150 iterations, 16 explored
replacements per population member, a gradient candidate set of 256, and seeds
0 through 4 for each direction. On CUDA, gradient microbatches are 4 for most
small checkpoints, 2 for Qwen 3 8B, and 1 for Gemma 3 12B. On Apple silicon,
Qwen 3 8B and Gemma 3 12B use search batch 1; Qwen 3 8B direction fitting uses
batch 2. Context graphs are backpropagated and released one at a time, then
averaged into the same 16 context objective.
Direction fitting uses the same model specific microbatch and a 768 token
prompt limit. Microbatch size does not change the candidate count. Validation
and test controllability records are not used during prompt search.

The C-MAP query budget is applied independently to each seed. The full protocol
therefore permits up to 512 validation discovery queries per seed, while the
scaling protocol permits up to 192. `cmap_queries.csv`, `cmap_directions.csv`,
and `cmap_summary.csv` retain the seed column. Prompt export writes
`optimized_prompt_controls.txt.provenance.json` with the source optimizer seed,
method, objective, and fluency score for every selected suffix.

The main reachable state study uses up to 512 prompt variants across the fixed
three way split. Final geometry, control, transfer, and causal tables use test
records only. Monitor fitting uses train records, validation chooses the
decision threshold, and test records provide the reported evaluation.
Override `EXAMPLE_LIMIT` only before a new run directory is created.
`CONTEXT_COUNT` controls the prompt optimization context count; the reported
study keeps it at 16.

## 4. Model input rules

The adapter records its choices in each study specification. Pretrained
checkpoints receive plain text. Instruction tuned checkpoints receive the
checkpoint's native chat template. Optimized prompt tokens are inserted inside
the user message before the generation marker, not as an assistant prefill.

Gemma 3 is loaded with its multimodal wrapper and the nested text transformer
is used for residual hooks. The experiments remain text only. Qwen 3 uses its
native chat template with thinking disabled so generated reasoning traces do
not create a hidden difference in output length or control cost across sizes.

## 5. Token pooling monitor study

The main archive stores pooled states because full token trajectories are
large. Run the token monitor study on the selected target layer and a fixed 64
example subset for all five primary checkpoints:

```bash
bash scripts/run_recommended_token_monitors.sh
```

This comparison trains last token, mean, max, and learned attention pooling
monitors on the same packed token state archive. The attention monitor uses
minibatches on CUDA. Keep the 64 example limit fixed across checkpoints.

## 6. Gemma Scope 2 feature study

After the corresponding reachable archives exist, run:

```bash
bash scripts/run_gemma_scope_matrix.sh
```

This study covers the instruction tuned Gemma 3 4B and 12B checkpoints. It
selects the fitted target layer from the direction sweep, loads the
official 16k all layer post residual SAE at the small sparsity setting, and
encodes only states from that model and layer. The output contains per sample
top features, reconstruction statistics, channel displacements from the
matching example baseline, effective ranks, and prompt to activation subspace
overlap in sparse feature space. It also trains a linear SAE feature monitor
with and without reachable state augmentation, tunes its threshold on the
validation split, and evaluates it on test states.

Required feature artifacts are:

```text
runs/controllability/<slug>/gemma_scope/feature_samples.jsonl
runs/controllability/<slug>/gemma_scope/feature_summary.csv
runs/controllability/<slug>/gemma_scope/feature_geometry.json
runs/controllability/<slug>/gemma_scope/sae_monitor_scores.csv
runs/controllability/<slug>/gemma_scope/sae_monitor_invariance.csv
runs/controllability/<slug>/gemma_scope/manifest.json
```

The feature geometry is a secondary mechanistic analysis. It does not replace
the residual geometry, and truncated top feature vectors should not be
described as a complete causal decomposition.

## 7. Study specification

`build-study-spec` reads the full layer sweep, chooses the largest signed
held out fitting gap, and creates an orthogonal random direction with the run
seed. `--layers sweep` captures five depth spaced layers plus the selected
layer. `--layers target` captures only the selected layer.

The generated JSON is the experiment contract. Review it before collection.
In particular, confirm:

```text
model.name
model.revision
model.dtype
model.family
model.loader
model.prompt_format
model.enable_thinking
data.path
data.limit
layers
target_directions
constraints
interventions
cmap
seed
```

Do not change a spec after a run begins. Write a new spec and output directory
for any revised threshold or intervention grid.

## 8. Core artifacts

For model slug `gemma3_4b_it`, the required files are:

```text
runs/controllability/gemma3_4b_it/directions/gemma3_4b_it_eval_awareness_layer_sweep.csv
runs/controllability/gemma3_4b_it/prompt_search/candidates.csv
runs/controllability/gemma3_4b_it/prompt_search/summary.csv
runs/controllability/gemma3_4b_it/optimized_prompt_controls.txt
runs/controllability/gemma3_4b_it/optimized_prompt_controls.txt.provenance.json
runs/controllability/gemma3_4b_it/study.json
runs/controllability/gemma3_4b_it/reachable/states.npz
runs/controllability/gemma3_4b_it/reachable/samples.jsonl
runs/controllability/gemma3_4b_it/reachable/geometry.csv
runs/controllability/gemma3_4b_it/reachable/trajectory_geometry.csv
runs/controllability/gemma3_4b_it/reachable/target_geometry.csv
runs/controllability/gemma3_4b_it/reachable/budget_growth.csv
runs/controllability/gemma3_4b_it/reachable/layer_propagation.csv
runs/controllability/gemma3_4b_it/reachable/principal_angles.csv
runs/controllability/gemma3_4b_it/reachable/split_half_stability.csv
runs/controllability/gemma3_4b_it/reachable/split_half_stability_summary.csv
runs/controllability/gemma3_4b_it/reachable/controllability_boundaries.csv
runs/controllability/gemma3_4b_it/reachable/controllability_boundary_summary.csv
runs/controllability/gemma3_4b_it/reachable/directed_accessibility.csv
runs/controllability/gemma3_4b_it/reachable/directed_accessibility_summary.csv
runs/controllability/gemma3_4b_it/reachable/detection_control_gap.csv
runs/controllability/gemma3_4b_it/reachable/representation_control_gap.csv
runs/controllability/gemma3_4b_it/reachable/controllability_atlas.csv
runs/controllability/gemma3_4b_it/reachable/boundary_survival.csv
runs/controllability/gemma3_4b_it/reachable/phase_transition_candidates.csv
runs/controllability/gemma3_4b_it/reachable/cmap/states.npz
runs/controllability/gemma3_4b_it/reachable/cmap/samples.jsonl
runs/controllability/gemma3_4b_it/reachable/cmap_directions.npz
runs/controllability/gemma3_4b_it/reachable/cmap_queries.csv
runs/controllability/gemma3_4b_it/reachable/cmap_directions.csv
runs/controllability/gemma3_4b_it/reachable/cmap_summary.csv
runs/controllability/gemma3_4b_it/control/control_costs.csv
runs/controllability/gemma3_4b_it/control/control_summary.csv
runs/controllability/gemma3_4b_it/control/dose_response.csv
runs/controllability/gemma3_4b_it/control/tracking_stability.csv
runs/controllability/gemma3_4b_it/jacobians.csv
runs/controllability/gemma3_4b_it/jacobians_spectrum.csv
runs/controllability/gemma3_4b_it/jacobians_control_basis_seed_0.npy
runs/controllability/gemma3_4b_it/jacobians_control_basis_seed_1.npy
runs/controllability/gemma3_4b_it/jacobians_control_basis_seed_2.npy
runs/controllability/gemma3_4b_it/jacobians_control_basis_seed_3.npy
runs/controllability/gemma3_4b_it/jacobians_control_basis_seed_4.npy
runs/controllability/gemma3_4b_it/jacobians.manifest.json
runs/controllability/gemma3_4b_it/transfer_source.csv
runs/controllability/gemma3_4b_it/transfer_source.summary.json
runs/controllability/gemma3_4b_it/transfer_category.csv
runs/controllability/gemma3_4b_it/transfer_category.summary.json
runs/controllability/gemma3_4b_it/causal/patching.csv
runs/controllability/gemma3_4b_it/causal/component_ablation.csv
runs/controllability/gemma3_4b_it/causal/head_ablation.csv
runs/controllability/gemma3_4b_it/causal/path_mediation.csv
runs/controllability/gemma3_4b_it/causal/cross_prompt_patching.csv
runs/controllability/gemma3_4b_it/monitors/monitor_scores.csv
runs/controllability/gemma3_4b_it/monitors/monitor_invariance.csv
runs/controllability/gemma3_4b_it/monitors/monitor_comparisons.csv
runs/controllability/gemma3_4b_it/monitors/ood_scores.csv
runs/controllability/gemma3_4b_it/monitors/manifest.json
runs/controllability/gemma3_4b_it/figures/manifest.json
```

After all model runs finish, the strict matrix aggregate must also contain:

```text
runs/controllability/matrix/manifest.json
runs/controllability/matrix/model_summaries.csv
runs/controllability/matrix/matched_contrasts.csv
runs/controllability/matrix/revisions.csv
runs/controllability/matrix/scaling_diagnostics.csv
runs/controllability/matrix/scaling_replication.csv
runs/controllability/matrix/figures/controllability_scaling.png
runs/controllability/matrix/figures/representation_control_gap.png
```

An experiment is incomplete if any required table is empty, if an expected
seed is missing, or if `samples.jsonl` points to a different model or spec.

## 9. Primary analyses

Reachability uses displacement from the matching example baseline. Pooling raw
states across questions would mostly measure question identity. Report
numerical rank, entropy effective rank, participation ratio, mean and maximum
displacement, preservation rate, prompt to activation subspace overlap, and
radius graph connectivity. The connectivity radius is the median nearest
neighbor displacement distance within the reported channel. Ordered activation
strength, ablation fraction, and controller setpoint sweeps define the
trajectory curvature table. `target_geometry.csv` separates displacement along
the fitted direction from orthogonal movement. `budget_growth.csv` recomputes
rank and radius at fixed cost thresholds, and `layer_propagation.csv` records
expansion or contraction between captured layers.

The within channel reference uses ten example grouped split half repeats for
the prompt and activation channels. Each half is reduced to at most 32 states
before its basis is estimated. This fixes the SVD workload and keeps the
reference sample count equal across the two halves.

`controllability_boundaries.csv` treats activation strength, ablation fraction,
and controller setpoint as ordered sweeps. It reports the largest preserved
control and the first observed failure beyond it. A sweep with no observed
failure is right censored. Optimized prompts are an unordered discrete sample,
so their envelope is labeled sample limited and must not be called a boundary.
The summary table bootstraps examples and preserves the censoring rate.

`directed_accessibility.csv` asks how closely states sampled through one
channel can be matched by states sampled through the other. Prompt and
activation sets are repeatedly reduced to the same number of points before
nearest set distance, directed Hausdorff distance, and coverage are measured.
This prevents a denser intervention grid from receiving an automatic
advantage. Report both directions because prompt to activation coverage need
not equal activation to prompt coverage. Examples with no accepted target state
remain in the table with zero coverage and contribute to the reported empty set
rate; they are not silently dropped from the normalized distance estimate.

`representation_control_gap.csv` places natural class separation and the mean
maximum behavior preserving movement on the same pooled projection scale. Its
gap is the standardized detection margin minus the standardized control
margin. `detection_control_gap.csv` retains the raw AUROC and span ratio for
backward compatible analysis. A positive gap is evidence that the fitted
feature is easier to detect than to control under the declared interventions.
It is not an impossibility result. Labeled examples with no accepted control
contribute zero movement.

C-MAP is fit only on validation examples. At each round it estimates the
accepted displacement tangent with an SVD, samples a fixed candidate pool,
and selects the residual direction with the largest combination of unexplored
tangent fraction and angular separation from earlier queries. Geometric dose
expansion and bisection estimate the behavior boundary. Accepted directions
are then frozen and evaluated on test examples. `cmap_queries.csv` retains
failed as well as accepted model executions, `cmap_directions.npz` stores the
actual control vectors, and `cmap_summary.csv` separates validation discovery
from held out evaluation. The stopping reason must be reported with rank and
radius; a finite budget map is not the full mathematical manifold.

`controllability_atlas.csv` reports preservation, accepted state rank, and
maximum displacement by concept, task category, source, layer, and channel.
It includes failed interventions in its denominator. The atlas describes the
declared controls and behavior gates. An empty or weak cell is not evidence
that no possible intervention can control that behavior.

`boundary_survival.csv` aligns each numeric intervention sweep to its sampled
dose range and bootstraps examples. `phase_transition_candidates.csv` marks an
abrupt boundary only when at least eight examples support a preservation drop
of at least 0.25 with bootstrap probability at least 0.95. The label is
`sharp_boundary_candidate`, not phase transition. A physical phase transition
claim would require denser local sweeps, schedule robustness, and replication.

The scaling aggregate fits predeclared Qwen 3 size trends for reachable rank,
boundary displacement, the representation control gap, and local Jacobian authority.
Three size points remain exploratory. Four points must have stable leave one
size out slope sign and an R squared value of at least 0.80 before receiving a
within family candidate label. The 0.6B, 1.7B, 4B, and 8B Qwen series can
support a Qwen-family scaling candidate. A general LLM scaling law still
requires the same slope direction in an independent family.

The Jacobian study constructs five independently seeded nested 32 dimensional
orthonormal residual control bases. The fitted concept direction is the first
coordinate, the declared orthogonal control is second, and seeded orthogonal directions fill
the remaining coordinates. Central finite differences with step 0.25 estimate
the last token state Jacobian at every captured layer for 16 fixed test
examples. Report results at dimensions 8, 16, and 32, including rank fraction,
the singular value spectrum, condition number, squared gain, and the Gramian
pseudodeterminant. Stability across basis sizes is required before discussing
local dimensionality. This remains a local result for direct residual control,
not a claim of full hidden state or prompt controllability.

Transfer pairs benchmark held out sources and subject categories within each
model, layer, and control channel. The analysis compares shared intervention
effects across groups, then tests whether transfer correlation rises with
principal angle overlap. Report every group pair, pooled Pearson and Spearman
association, and the standardized overlap coefficient after controlling for
displacement, preservation rate, layer depth, and shared intervention count.
Each summary also reports these statistics separately for prompt, activation,
hybrid, and random channels.

Monitor evaluation repeats every declared monitor fit for seeds 0 through 4
and preserves seed-level score, invariance, OOD, and comparison tables. It
follows the declared question level train, validation, and test split. The
neutral and evaluation versions of a question always remain in
the same partition. Train the natural monitor on baseline train states only.
Train the augmented monitor on baseline plus behavior preserving train states.
Choose each decision threshold on baseline validation states, then evaluate
once on test states. Report average agreement, worst case consistency, score
drift, accuracy, and latent OOD distance. A permuted label linear probe is the
random probe control. `monitor_comparisons.csv` reports paired effect sizes,
bootstrap intervals, sign permutation tests, Holm adjusted values, and
Benjamini Hochberg false discovery values.

Control energy uses the recorded prompt token count, activation norm, adaptive
controller energy, or Euclidean prompt plus activation cost. Report failure to
reach a setpoint rather than replacing it with the nearest state.
The orthogonal random direction remains a negative control and is excluded
from the minimum cost envelope. `dose_response.csv` reports slope, fit, and
monotonic fraction. `tracking_stability.csv` reports generation step error,
maximum error, final error, and update size for the adaptive controller.

## 10. Causal route tests

The model script runs `causal-study` after reachable state collection. It
selects one behavior preserving optimized prompt and fixed activation control
per question by minimizing their target projection gap, then deterministically
keeps the closest matched examples up to the declared limit. It reruns both controls
deterministically, patches the source last token state into the other channel at
every captured layer, and repeats the patch in the reverse direction.

`patching.csv` reports logit recovery, patch effect, Jensen Shannon divergence,
and top token changes. `component_ablation.csv` reports the effect of zeroing
the attention output or MLP output projection at each captured layer under both
matched controls. `head_ablation.csv` tests a fixed depth spaced sample of
individual attention heads. `path_mediation.csv` patches a sender state and
then restores the next captured receiver attention or MLP output to its target
reference. The difference estimates how much of the sender patch effect passes
through that receiver. `cross_prompt_patching.csv` performs both patch
directions between the neutral and evaluation versions of the same held out
question. These are causal route tests; cosine similarity alone does not
establish mechanism reuse.

## 11. Statistical rules

Use paired bootstrap intervals over matched examples and paired permutation
tests for method comparisons. Keep source, concept, model, and layer visible in
the analysis table. Search, C-MAP, monitor, and Jacobian seeds measure
stochastic variation in their respective procedures. They do not replace
example level uncertainty or model family replication.
Report paired standardized effects. Apply Holm correction to the confirmatory
comparison family and Benjamini Hochberg correction to the corresponding
exploratory table. Validation chooses layers, monitor thresholds, and controller
settings. Test records are evaluated once under the fixed specification.

Before any claim enters the paper:

* all five full checkpoints must have complete full protocol artifacts
* all four reduced Qwen runs must use the same seeds, contexts, examples, and C-MAP budget
* every stochastic headline table must retain all five declared seed values
* every geometry row must have a matching behavior preservation count
* ordered boundary claims must report bracketed and censored rates
* prompt envelopes must remain labeled as unordered finite samples
* cross channel accessibility must use equal count subsampling and both directions
* detection and control must be reported in common units and as separate quantities
* C-MAP direction selection must use validation only and report held out test rows
* C-MAP must retain failed queries, stopping reasons, and boundary censoring
* atlas cells must retain failed interventions in preservation denominators
* sharp boundary claims must satisfy the registered support and bootstrap rules
* only the four point Qwen series may be used for the declared size fit
* a general scaling claim requires an independently replicated family trend
* transfer must use held out source groups
* monitors must use question grouped train, validation, and test partitions
* random directions must be orthogonal to the fitted concept direction
* token pooling comparisons must use the same token archive
* causal claims must come from patching, receiver blocking, or ablation, not projection changes
* Gemma Scope claims must report SAE reconstruction error and retained feature count

## 12. What remains after running the scripts

The code path, commands, gates, metrics, and artifact formats are implemented.
The remaining work is empirical: complete the primary matrix and Qwen scaling
tier, inspect failed behavior gates and C-MAP stopping diagnostics, run the
token monitor and Gemma Scope subsets, and place the generated tables and
figures into the paper. No result should be written from terminal output or an
untracked plot.
