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
uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name())"
git rev-parse HEAD
```

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

The matrix has two tiers. Five checkpoints receive the complete protocol:

| Role | Hugging Face checkpoint | Slug |
| --- | --- | --- |
| Small Gemma anchor | `google/gemma-3-4b-it` | `gemma3_4b_it` |
| Mid scale Gemma anchor | `google/gemma-3-12b-it` | `gemma3_12b_it` |
| Gemma generation comparison | `google/gemma-4-12B-it` | `gemma4_12b_it` |
| Small cross family replication | `microsoft/Phi-4-mini-instruct` | `phi4_mini_instruct` |
| Mid scale cross family replication | `microsoft/phi-4` | `phi4` |

Four more checkpoints complete the matched post training comparisons:

| Comparison | Hugging Face checkpoint | Slug |
| --- | --- | --- |
| Gemma 3 4B PT versus IT | `google/gemma-3-4b-pt` | `gemma3_4b_pt` |
| Gemma 3 12B PT versus IT | `google/gemma-3-12b-pt` | `gemma3_12b_pt` |
| Gemma 4 12B PT versus IT | `google/gemma-4-12B` | `gemma4_12b_pt` |
| Phi 4 versus reasoning post training | `microsoft/Phi-4-reasoning` | `phi4_reasoning` |

Run all nine checkpoints in the declared order:

```bash
bash scripts/run_recommended_matrix.sh
```

The source of truth for membership, family labels, protocol tier, and matched
comparisons is `configs/recommended_matrix.json`. It also declares precision,
attention implementation, and which downstream analyses apply to each model.
The script finishes with
`aggregate-study-matrix`, which rejects a missing or empty required artifact
before writing combined tables and matched post training contrasts.

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

The final argument is `full` or `matched`. Both protocols use the same five
seeds, search iterations, context count, behavior gates, reachable state
collection, and downstream analyses. The matched protocol runs EPO and GCG but
omits random search, uniform random prompts, and natural text scanning as
optimizer comparisons. Natural prompt interventions remain in the reachable
state study.

The Gemma 3 4B to 12B pair supports a within family scale comparison. Gemma 3
12B and Gemma 4 12B support a cross generation comparison, but not an isolated
architecture claim because their training data and recipes also differ. Phi 4
Mini and Phi 4 provide an independent family replication at two scales.
Pretrained and instruction tuned Gemma pairs, and the Phi 4 to Phi 4 Reasoning
pair, are the matched post training analyses.

Each script performs the following work without a smoke run substitution:

1. It fits the paired direction at every transformer layer.
2. It retains the strongest layer and runs EPO, GCG, equal budget random
   search, uniform random prompts, and natural text scanning in both projection
   directions.
3. It exports eight decrease and eight increase controls from the two Pareto
   sets.
4. It creates fixed addition, directional ablation, PID, orthogonal random,
   per example prompt rewrite, optimized prompt, and hybrid sweeps.
5. It generates one deterministic output, then recaptures that exact sequence
   without leaving the intervention lifecycle. PID state remains live through
   capture, while generation cost and tracking error are snapshotted first.
6. It admits a state only when task correctness, prompt semantics, output
   semantics, output quality, and the control budget all pass.
7. It computes setpoint reachability, dose response, budget growth, layer
   propagation, source and subject transfer, local Jacobian spectra, causal
   route tests, monitor invariance, and reachable state augmentation.
8. It renders the predeclared figures directly from the archived tables.

The prompt search averages every objective over 16 training contexts sampled
per seed. It uses sequence length 32, population 24, 150 iterations, 16 explored
replacements per population member, a gradient candidate set of 256, and seeds
0 through 4 for each direction. Gradient microbatches are 4 for the 4B
checkpoints and 1 for the 12B to 14B checkpoints. Context graphs are backpropagated
and released one at a time, then averaged into the same 16 context objective.
Direction fitting uses the same model specific microbatch and a 768 token
prompt limit. Microbatch size does not change the candidate count. Validation
and test controllability records are not used during prompt search.

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

Gemma 4 is loaded with `AutoModelForMultimodalLM` and its nested text model is
used for residual hooks. The experiments remain text only. Thinking is disabled
for Gemma 4 so generated reasoning traces do not create a hidden
difference in output length or control cost. Phi 4 Reasoning keeps its native
reasoning behavior because that post training change is the object of the
matched comparison.

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

This study covers Gemma 3 4B and 12B in both pretrained and instruction tuned
forms. It selects the fitted target layer from the direction sweep, loads the
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
runs/controllability/gemma3_4b_it/study.json
runs/controllability/gemma3_4b_it/reachable/states.npz
runs/controllability/gemma3_4b_it/reachable/samples.jsonl
runs/controllability/gemma3_4b_it/reachable/geometry.csv
runs/controllability/gemma3_4b_it/reachable/trajectory_geometry.csv
runs/controllability/gemma3_4b_it/reachable/target_geometry.csv
runs/controllability/gemma3_4b_it/reachable/budget_growth.csv
runs/controllability/gemma3_4b_it/reachable/layer_propagation.csv
runs/controllability/gemma3_4b_it/reachable/principal_angles.csv
runs/controllability/gemma3_4b_it/control/control_costs.csv
runs/controllability/gemma3_4b_it/control/control_summary.csv
runs/controllability/gemma3_4b_it/control/dose_response.csv
runs/controllability/gemma3_4b_it/control/tracking_stability.csv
runs/controllability/gemma3_4b_it/jacobians.csv
runs/controllability/gemma3_4b_it/jacobians_spectrum.csv
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
runs/controllability/gemma3_4b_it/figures/manifest.json
```

After all model runs finish, the strict matrix aggregate must also contain:

```text
runs/controllability/matrix/manifest.json
runs/controllability/matrix/model_summaries.csv
runs/controllability/matrix/matched_contrasts.csv
runs/controllability/matrix/revisions.csv
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

The Jacobian study uses the fitted concept direction and its orthogonal random
control as a fixed two dimensional residual basis. Central finite differences
with step 0.25 estimate the last token state Jacobian at every captured layer
for 16 fixed test examples. Report rank, the full singular value spectrum,
condition number, and the Gramian pseudodeterminant. This is a local result for
the declared control basis, not a claim of full hidden state controllability.

Transfer pairs benchmark held out sources and subject categories within each
model, layer, and control channel. The analysis compares shared intervention
effects across groups, then tests whether transfer correlation rises with
principal angle overlap. Report every group pair, pooled Pearson and Spearman
association, and the standardized overlap coefficient after controlling for
displacement, preservation rate, layer depth, and shared intervention count.
Each summary also reports these statistics separately for prompt, activation,
hybrid, and random channels.

Monitor evaluation follows the declared question level train, validation, and
test split. The neutral and evaluation versions of a question always remain in
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
per question by minimizing their target projection gap. It reruns both controls
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
the analysis table. Search seeds measure optimizer variation. They do not
replace example level uncertainty or model family replication.
Report paired standardized effects. Apply Holm correction to the confirmatory
comparison family and Benjamini Hochberg correction to the corresponding
exploratory table. Validation chooses layers, monitor thresholds, and controller
settings. Test records are evaluated once under the fixed specification.

Before any claim enters the paper:

* all five full checkpoints must have complete full protocol artifacts
* all four extra matched checkpoints must have complete matched protocol artifacts
* every geometry row must have a matching behavior preservation count
* transfer must use held out source groups
* monitors must use question grouped train, validation, and test partitions
* random directions must be orthogonal to the fitted concept direction
* token pooling comparisons must use the same token archive
* causal claims must come from patching, receiver blocking, or ablation, not projection changes
* Gemma Scope claims must report SAE reconstruction error and retained feature count

## 12. What remains after running the scripts

The code path, commands, gates, metrics, and artifact formats are implemented.
The remaining work is empirical: complete the model matrix, inspect failed
behavior gates, run the token monitor and Gemma Scope subsets, and place the
generated tables and figures into the paper. No result should be written from
terminal output or an untracked plot.
