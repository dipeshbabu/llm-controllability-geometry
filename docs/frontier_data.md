# Frontier data sources

Use `tests/fixtures/` only for tests. Paper experiments should use local files
generated under `data/frontier/`.

## Primary sources

The data builder supports these sources:

| Source key | Dataset | Why it belongs |
| --- | --- | --- |
| `hle` | `cais/hle` | Expert level text questions with broad coverage. Rows requiring images or media are excluded. |
| `gpqa_diamond` | `Idavidrein/gpqa`, `gpqa_diamond` config | Graduate level biology, chemistry, and physics questions designed to be difficult even with web access. |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` | More difficult and less saturated than MMLU, with 10 answer options and broad subject coverage. |
| `math500` | `HuggingFaceH4/MATH-500` | Standard competition math control set with exact answers. |

`hle` and `gpqa_diamond` are gated. Accept the Hugging Face terms and run
`hf auth login` before building those files. Do not commit generated
benchmark rows or paste raw examples into public artifacts.

## Build command

```bash
uv sync --extra remote
uv run llm-controllability build-frontier-data \
  --out-dir data/frontier \
  --sources mmlu_pro math500 gpqa_diamond hle \
  --allow-gated \
  --max-items-per-source 500 \
  --train-fraction 0.60 \
  --validation-fraction 0.20 \
  --behavior-limit 300
```

The command writes:

- `data/frontier/text_pools/frontier_train.txt`
- `data/frontier/text_pools/frontier_validation.txt`
- `data/frontier/text_pools/frontier_test.txt`
- `data/frontier/contrasts/eval_awareness_train.json`
- `data/frontier/contrasts/eval_awareness_validation.json`
- `data/frontier/contrasts/eval_awareness_test.json`
- `data/frontier/contrasts/casing_control.json`
- `data/frontier/contrasts/style_control.json`
- `data/frontier/behavior/frontier_behavior.json`
- `data/frontier/behavior/controllability_train.jsonl`
- `data/frontier/behavior/controllability_validation.jsonl`
- `data/frontier/behavior/controllability_test.jsonl`
- `data/frontier/behavior/controllability_all.jsonl`
- `data/frontier/controls/eval_awareness_natural.txt`
- `data/frontier/controls/prompt_rewrites.json`

Use training contrasts for fitting residual directions and validation contrasts
for layer selection. Test examples remain untouched until reachable state
collection under a fixed specification. GPQA choices are
deterministically shuffled and their correct label is recomputed after
shuffling. Splits are performed within source. Paired neutral and evaluation
prompts share a `pair_id`, which keeps both variants in the same declared
train, validation, or test partition. The prompt rewrite map contains natural
paraphrase, concise style, capitalization, and topic matched controls for each
record.

The task gate uses explicit multiple choice extraction, signed numeric
comparison, or an extracted final answer. It does not accept generic answer
substring matches for the generated controllability records.

## Paper use

The benchmark questions are the task constraint, not the contribution. Paired
wrappers isolate evaluation context while leaving the question and answer
fixed. The analysis admits an intervened state only when the answer verifier,
prompt semantic gate, output semantic gate, output quality gate, and control
budget all pass.

Report source counts from `data/frontier/manifest.json`, not from memory.
