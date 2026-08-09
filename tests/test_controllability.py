import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from llm_controllability.causal import (
    ActivationCache,
    AttentionHeadAblation,
    StatePatching,
    trajectory_similarity,
)
from llm_controllability.constraints import (
    BehaviorGate,
    BehaviorRecord,
    BudgetConstraint,
    PromptSemanticEquivalenceConstraint,
    TaskPreservationConstraint,
)
from llm_controllability.constraints.verification import verify_output
from llm_controllability.controllability.study import build_interventions
from llm_controllability.controllability.types import (
    ControlChannel,
    InterventionMetadata,
    StateSample,
)
from llm_controllability.evaluation import monitor_invariance
from llm_controllability.evaluation.causal_study import (
    matched_control_pairs,
    matched_prompt_pairs,
)
from llm_controllability.evaluation.control_study import (
    control_cost_rows,
    summarize_control_costs,
)
from llm_controllability.evaluation.discovery import (
    scaling_diagnostic_rows,
    scaling_replication_rows,
)
from llm_controllability.evaluation.jacobian_study import finite_difference_jacobians
from llm_controllability.evaluation.matrix import _TABLES, aggregate_matrix
from llm_controllability.evaluation.monitor_study import (
    _partition_ids,
    _split_ids,
    paired_monitor_comparisons,
)
from llm_controllability.evaluation.statistics import adjust_pvalues
from llm_controllability.evaluation.transfer_study import transfer_rows
from llm_controllability.interventions import (
    ActivationAddition,
    AdaptiveActivationController,
    DirectionalAblation,
    MappedPromptIntervention,
)
from llm_controllability.monitors import (
    LinearMonitor,
    MahalanobisOODMonitor,
    pool_hidden_states,
)
from llm_controllability.optimization.contextual import ContextualTargetRunner
from llm_controllability.optimization.epo import token_grads
from llm_controllability.reachability import (
    ActiveTangentExplorer,
    CMapConfig,
    adaptive_control_boundary,
    boundary_survival_rows,
    control_jacobian,
    controllability_atlas_rows,
    controllability_boundary_rows,
    detection_control_gap_rows,
    directed_accessibility_rows,
    discover_controllability_manifold,
    effective_rank,
    load_state_samples,
    local_controllability,
    phase_transition_candidate_rows,
    principal_angles,
    save_state_samples,
    split_half_stability_rows,
    subspace_overlap,
    summarize_controllability_boundaries,
    summarize_directed_accessibility,
    summarize_split_half_stability,
    summarize_trajectories,
)
from llm_controllability.reachability.collection import run_and_capture
from llm_controllability.reachability.geometry import (
    baseline_displacements,
    budget_growth,
    layerwise_propagation,
    summarize_reachability,
    target_orthogonal_decomposition,
)


class DummyBlock(torch.nn.Module):
    def forward(self, hidden):
        return (hidden + 1.0,)


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyBlock(), DummyBlock()])


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyBackbone()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, hidden):
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return hidden


class ReplayBatch(dict):
    def to(self, device):
        return ReplayBatch({key: value.to(device) for key, value in self.items()})


class ReplayTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 9

    def __call__(self, text, **kwargs):
        values = torch.tensor([[1, 2]], dtype=torch.long)
        return ReplayBatch(
            input_ids=values,
            attention_mask=torch.ones_like(values),
        )

    @staticmethod
    def encode(text, add_special_tokens=False):
        return [1, 2] if text == "question" else [3, 4]

    @staticmethod
    def decode(ids, skip_special_tokens=True):
        return "answer"


class ReplayModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 2)
        torch.nn.init.zeros_(self.embedding.weight)
        self.model = DummyBackbone()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        hidden = self.model.layers[0](hidden)[0]
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.embedding.num_embeddings,
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits, hidden=hidden)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        generated = input_ids
        for token in (3, 4):
            self(
                input_ids=generated,
                attention_mask=torch.ones_like(generated),
            )
            next_token = torch.full(
                (generated.shape[0], 1),
                token,
                dtype=torch.long,
                device=generated.device,
            )
            generated = torch.cat((generated, next_token), dim=1)
        return generated


def make_sample(
    example_id,
    layer,
    channel,
    state,
    *,
    name=None,
    preserved=True,
    label=1.0,
    tags=None,
    target_projection=None,
    parameters=None,
    control_cost=None,
    constraint_results=None,
):
    metrics = {"monitor_label": label}
    if target_projection is not None:
        metrics["target_projection"] = target_projection
    return StateSample(
        example_id=example_id,
        model_name="dummy",
        layer=layer,
        intervention=InterventionMetadata(
            name=name or channel.value,
            channel=channel,
            control_cost=(
                0.0
                if channel is ControlChannel.BASELINE
                else 1.0
                if control_cost is None
                else control_cost
            ),
            parameters=parameters or {},
        ),
        state=np.asarray(state, dtype=np.float32),
        prompt="p",
        output="o",
        behavior_preserved=preserved,
        constraint_results=constraint_results or {},
        metrics=metrics,
        tags=tags or {},
    )


class ControllabilityTests(unittest.TestCase):
    def test_cmap_runs_live_discovery_then_held_out_evaluation(self):
        model = ReplayModel()
        tokenizer = ReplayTokenizer()
        gate = BehaviorGate(
            [
                TaskPreservationConstraint(require_correct=True),
                BudgetConstraint(2.0),
            ]
        )
        example = {
            "prompt": "question",
            "answer": "answer",
            "verifier": "exact",
            "monitor_label": 1,
        }

        result = discover_controllability_manifold(
            model,
            tokenizer,
            model_name="dummy",
            validation_examples=[{"id": "validation", "split": "validation", **example}],
            test_examples=[{"id": "test", "split": "test", **example}],
            behavior_gate=gate,
            generation={"max_new_tokens": 2},
            config=CMapConfig(
                layer=0,
                direction_budget=1,
                query_budget=2,
                candidate_pool_size=8,
                initial_strength=0.5,
                maximum_strength=0.5,
                validation_examples=1,
                test_examples=1,
                seed=3,
            ),
            target_direction=np.asarray([1.0, 0.0]),
        )

        self.assertEqual(len(result.directions), 1)
        self.assertEqual({row["role"] for row in result.summary_rows}, {"validation", "test"})
        self.assertEqual({row["role"] for row in result.query_rows}, {"validation", "test"})
        self.assertTrue(all(sample.behavior_preserved for sample in result.samples))

    def test_cmap_expands_unexplored_tangent_and_brackets_boundary(self):
        config = CMapConfig(
            layer=0,
            direction_budget=3,
            query_budget=16,
            candidate_pool_size=32,
            validation_examples=1,
            test_examples=1,
            seed=7,
        )
        explorer = ActiveTangentExplorer(4, config, seed=7)
        first = explorer.propose()
        self.assertIsNotNone(first)
        first_direction, _, _ = first
        explorer.observe(first_direction, preserved=True)
        explorer.finish_direction(0)

        second = explorer.propose()
        self.assertIsNotNone(second)
        second_direction, novelty, _ = second

        self.assertAlmostEqual(float(np.dot(first_direction, second_direction)), 0.0, places=5)
        self.assertGreater(novelty, 0.9)

        boundary = adaptive_control_boundary(
            lambda strength: float(strength <= 3.0),
            initial_strength=1.0,
            maximum_strength=8.0,
            expansion_factor=2.0,
            boundary_steps=2,
            required_preservation_rate=1.0,
            max_trials=8,
        )
        self.assertEqual(boundary.status, "bracketed")
        self.assertEqual(boundary.lower, 3.0)
        self.assertEqual(boundary.upper, 3.5)

    def test_matrix_aggregation_requires_and_combines_declared_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "slug": slug,
                                "model_name": slug,
                                "protocol": "full",
                                "family": "dummy",
                                "scale_group": "small",
                                "parameter_count_billions": 1.0,
                                "training_regime": "instruction",
                            }
                            for slug in ("reference", "comparison")
                        ],
                        "matched_comparisons": [
                            {
                                "name": "matched",
                                "reference": "reference",
                                "comparison": "comparison",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows_by_table = {
                "geometry": {
                    "channel": "prompt",
                    "preservation_rate": 1.0,
                    "effective_rank": 1.0,
                    "participation_ratio": 1.0,
                    "mean_displacement": 1.0,
                },
                "control": {
                    "channel": "prompt",
                    "reach_rate": 1.0,
                    "mean_minimum_cost": 1.0,
                    "mean_control_error": 0.0,
                },
                "controllability_boundaries": {
                    "channel": "activation",
                    "family": "activation_addition",
                    "side": "increase",
                    "feasible_example_rate": 1.0,
                    "bracketed_rate": 1.0,
                    "mean_lower_control_bound": 2.0,
                    "mean_maximum_state_displacement": 3.0,
                },
                "directed_accessibility": {
                    "direction": "prompt_to_activation",
                    "mean_normalized_gap": 0.2,
                    "mean_coverage_rate": 0.8,
                    "mean_directed_hausdorff_normalized": 0.4,
                    "estimable_example_rate": 1.0,
                    "target_empty_rate": 0.0,
                },
                "detection_control_gap": {
                    "channel": "activation",
                    "natural_oriented_projection_auc": 1.0,
                    "control_to_detection_ratio": 0.5,
                    "controlled_example_rate": 1.0,
                    "low_control_fraction": 0.25,
                },
                "representation_control_gap": {
                    "channel": "activation",
                    "natural_oriented_projection_auc": 1.0,
                    "standardized_detection_margin": 2.0,
                    "standardized_control_margin": 0.5,
                    "representation_control_gap": 1.5,
                },
                "split_half_stability": {
                    "channel": "prompt",
                    "mean_subspace_overlap": 1.0,
                    "mean_effective_rank": 1.0,
                },
                "controllability_atlas": {
                    "channel": "prompt",
                    "facet": "concept",
                    "facet_value": "test",
                    "preservation_rate": 1.0,
                    "controlled_example_rate": 1.0,
                    "effective_rank": 1.0,
                    "maximum_state_displacement": 1.0,
                },
                "boundary_survival": {
                    "channel": "activation",
                    "dose_fraction": 1.0,
                    "preservation_rate": 0.0,
                },
                "phase_transition_candidates": {
                    "channel": "activation",
                    "family": "activation_addition",
                    "side": "increase",
                    "largest_preservation_drop": 1.0,
                    "final_preservation_rate": 0.0,
                    "sharp_boundary_candidate": 1,
                },
                "cmap_summary": {
                    "role": "test",
                    "preservation_rate": 1.0,
                    "effective_rank": 2.0,
                    "participation_ratio": 1.8,
                    "maximum_state_displacement": 3.0,
                    "mean_boundary_lower": 2.0,
                },
                "jacobians": {
                    "control_dimension": 8,
                    "rank_fraction": 1.0,
                    "maximum_gain": 1.0,
                    "minimum_nonzero_gain": 1.0,
                    "squared_gain": 8.0,
                },
                "monitor_invariance": {
                    "monitor": "linear",
                    "training": "natural",
                    "channel": "prompt",
                    "worst_case_consistency": 1.0,
                    "accuracy": 1.0,
                    "maximum_score_drift": 0.0,
                },
                "patching": {
                    "direction": "prompt_to_activation",
                    "logit_recovery": 1.0,
                    "patched_source_js": 0.0,
                },
            }
            for slug in ("reference", "comparison"):
                run_dir = root / "runs" / slug
                manifest = run_dir / "reachable" / "manifest.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text(
                    json.dumps({"runtime": {"model_revision": "abc"}}),
                    encoding="utf-8",
                )
                for table_name, relative in _TABLES.items():
                    path = run_dir / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    row = rows_by_table.get(
                        table_name,
                        {"value": 1.0},
                    )
                    with path.open(
                        "w",
                        encoding="utf-8",
                        newline="",
                    ) as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)

            manifest = aggregate_matrix(
                root / "runs",
                matrix_path,
                root / "aggregate",
            )

            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["n_models"], 2)
            self.assertTrue((root / "aggregate" / "matched_contrasts.csv").exists())
            self.assertTrue((root / "aggregate" / "scaling_diagnostics.csv").exists())

    def test_atlas_preserves_behavior_failures_in_coverage_rates(self):
        samples = [
            make_sample(
                "a",
                0,
                ControlChannel.BASELINE,
                [0.0, 0.0],
                tags={"concept": "truthfulness"},
                target_projection=0.0,
            ),
            make_sample(
                "a",
                0,
                ControlChannel.ACTIVATION,
                [1.0, 0.0],
                tags={"concept": "truthfulness"},
                target_projection=1.0,
                preserved=True,
            ),
            make_sample(
                "b",
                0,
                ControlChannel.BASELINE,
                [0.0, 0.0],
                tags={"concept": "truthfulness"},
                target_projection=0.0,
            ),
            make_sample(
                "b",
                0,
                ControlChannel.ACTIVATION,
                [2.0, 0.0],
                tags={"concept": "truthfulness"},
                target_projection=2.0,
                preserved=False,
            ),
        ]

        rows = controllability_atlas_rows(samples)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["facet_value"], "truthfulness")
        self.assertEqual(rows[0]["controlled_example_rate"], 0.5)
        self.assertEqual(rows[0]["maximum_state_displacement"], 1.0)

    def test_phase_transition_requires_a_supported_sharp_drop(self):
        samples = []
        for index in range(10):
            example_id = f"e{index}"
            samples.append(
                make_sample(
                    example_id,
                    0,
                    ControlChannel.BASELINE,
                    [0.0, 0.0],
                    target_projection=0.0,
                )
            )
            for strength in (1.0, 2.0, 4.0, 8.0):
                samples.append(
                    make_sample(
                        example_id,
                        0,
                        ControlChannel.ACTIVATION,
                        [strength, 0.0],
                        name=f"activation_addition_a{strength:g}",
                        target_projection=strength,
                        parameters={"strength": strength},
                        preserved=strength < 4.0,
                    )
                )

        survival = boundary_survival_rows(samples, bootstrap_resamples=100)
        candidates = phase_transition_candidate_rows(survival)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["qualification"], "sharp_boundary_candidate")
        self.assertEqual(candidates[0]["largest_preservation_drop"], 1.0)

    def test_scaling_diagnostics_require_four_stable_scale_points(self):
        models = []
        summaries = []
        for family in ("family_a", "family_b"):
            for size in (1.0, 2.0, 4.0, 8.0):
                slug = f"{family}_{size:g}"
                models.append(
                    {
                        "slug": slug,
                        "family": family,
                        "training_regime": "instruction",
                        "parameter_count_billions": size,
                    }
                )
                summaries.append(
                    {
                        "slug": slug,
                        "table": "geometry",
                        "metric": "effective_rank",
                        "channel": "activation",
                        "mean": 2.0 * size,
                    }
                )

        diagnostics = scaling_diagnostic_rows(summaries, models)
        replication = scaling_replication_rows(diagnostics)

        self.assertEqual(len(diagnostics), 2)
        self.assertTrue(
            all(
                row["qualification"] == "within_family_scaling_candidate"
                for row in diagnostics
            )
        )
        self.assertAlmostEqual(diagnostics[0]["scaling_exponent_or_slope"], 1.0)
        self.assertEqual(replication[0]["qualification"], "replicated_scaling_candidate")

    def test_behavior_gate_is_a_hard_conjunction(self):
        gate = BehaviorGate(
            [
                TaskPreservationConstraint(require_correct=True),
                BudgetConstraint(maximum_cost=2.0),
            ]
        )
        reference = BehaviorRecord("a", task_score=1.0, task_correct=True)
        candidate = BehaviorRecord("a", task_score=1.0, task_correct=True)

        passed, results = gate.evaluate(reference, candidate, control_cost=3.0)

        self.assertFalse(passed)
        self.assertTrue(results["task"].passed)
        self.assertFalse(results["control_budget"].passed)

    def test_task_gate_rejects_an_incorrect_reference(self):
        constraint = TaskPreservationConstraint(require_correct=True)
        reference = BehaviorRecord("wrong", task_score=0.0, task_correct=False)
        candidate = BehaviorRecord("right", task_score=1.0, task_correct=True)

        result = constraint.evaluate(reference, candidate, control_cost=0.0)

        self.assertFalse(result.passed)
        self.assertIn("reference_correct=False", result.details)

    def test_prompt_semantic_gate_reads_prompt_embeddings(self):
        constraint = PromptSemanticEquivalenceConstraint(minimum_similarity=0.9)
        reference = BehaviorRecord(
            "answer",
            metadata={"prompt_embedding": np.array([1.0, 0.0])},
        )
        matching = BehaviorRecord(
            "answer",
            metadata={"prompt_embedding": np.array([0.99, 0.01])},
        )
        changed = BehaviorRecord(
            "answer",
            metadata={"prompt_embedding": np.array([0.0, 1.0])},
        )

        self.assertTrue(
            constraint.evaluate(reference, matching, control_cost=1.0).passed
        )
        self.assertFalse(
            constraint.evaluate(reference, changed, control_cost=1.0).passed
        )

    def test_task_verifiers_preserve_sign_and_extract_final_answers(self):
        self.assertEqual(
            verify_output("Final answer: -2", {"answer": "-2", "verifier": "numeric"}),
            (1.0, True),
        )
        self.assertEqual(
            verify_output("Final answer: 2", {"answer": "-2", "verifier": "numeric"}),
            (0.0, False),
        )
        self.assertEqual(
            verify_output(
                "A is tempting. Final answer is (B).",
                {"answer": "B", "verifier": "multiple_choice"},
            ),
            (1.0, True),
        )
        self.assertEqual(
            verify_output(
                "Reasoning\n\\boxed{x+1}",
                {"answer": "x+1", "verifier": "final_answer"},
            ),
            (1.0, True),
        )

    def test_contextual_target_runner_averages_contexts_and_keeps_gradients(self):
        class ContextModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 2)
                with torch.no_grad():
                    self.embedding.weight.copy_(
                        torch.arange(16, dtype=torch.float32).reshape(8, 2)
                    )

            def get_input_embeddings(self):
                return self.embedding

        model = ContextModel()

        def base_runner(*, input_ids=None, inputs_embeds=None):
            hidden = (
                model.get_input_embeddings()(input_ids)
                if input_ids is not None
                else inputs_embeds
            )
            token_score = hidden.sum(dim=-1)
            return {
                "target": token_score.mean(dim=1),
                "logits": token_score.unsqueeze(-1).expand(-1, -1, 8),
            }

        context_parts = [
            (torch.tensor([1]), torch.tensor([0])),
            (torch.tensor([2, 3]), torch.tensor([1, 0])),
        ]
        runner = ContextualTargetRunner(
            base_runner,
            model,
            context_parts,
        )
        candidate_ids = torch.tensor([[4, 5], [6, 7]])
        result = runner(input_ids=candidate_ids)

        expected_targets = []
        for prefix, suffix in context_parts:
            combined = torch.cat(
                [
                    prefix.view(1, -1).expand(2, -1),
                    candidate_ids,
                    suffix.view(1, -1).expand(2, -1),
                ],
                dim=1,
            )
            expected_targets.append(base_runner(input_ids=combined)["target"])
        self.assertTrue(
            torch.allclose(result["target"], torch.stack(expected_targets).mean(dim=0))
        )
        self.assertEqual(result["logits"].shape, (2, 2, 8))
        self.assertTrue(
            torch.allclose(result["xentropy"], torch.full((2,), np.log(8.0)))
        )

        candidate_embeds = model.get_input_embeddings()(candidate_ids).detach()
        candidate_embeds.requires_grad = True
        runner(inputs_embeds=candidate_embeds)["target"].sum().backward()
        self.assertGreater(float(candidate_embeds.grad.abs().sum()), 0.0)

        state = token_grads(
            model,
            runner,
            candidate_ids,
            x_penalty=torch.zeros(candidate_ids.shape[0]),
            batch_size=1,
        )
        self.assertEqual(state.token_grads.shape, (2, 2, 8))
        self.assertTrue(torch.isfinite(state.token_grads).all())
        self.assertGreater(float(state.token_grads.abs().sum()), 0.0)

    def test_fixed_addition_and_ablation_modify_only_requested_direction(self):
        model = DummyModel()
        hidden = torch.zeros(1, 2, 3)
        addition = ActivationAddition("add", 0, torch.tensor([1.0, 0.0, 0.0]), 2.0)
        with addition.apply(model):
            output = model(hidden)
        self.assertTrue(torch.allclose(output[0, -1], torch.tensor([4.0, 2.0, 2.0])))
        self.assertTrue(torch.allclose(output[0, 0], torch.tensor([2.0, 2.0, 2.0])))

        ablation = DirectionalAblation(
            "ablate",
            0,
            torch.tensor([1.0, 0.0, 0.0]),
            token_scope="all",
        )
        with ablation.apply(model):
            output = model(hidden)
        self.assertTrue(torch.allclose(output[..., 0], torch.ones(1, 2)))
        self.assertTrue(torch.allclose(output[..., 1:], torch.full((1, 2, 2), 2.0)))

    def test_adaptive_controller_tracks_projection(self):
        model = DummyModel()
        controller = AdaptiveActivationController(
            "pid",
            0,
            torch.tensor([1.0, 0.0]),
            setpoint=3.0,
            kp=1.0,
            token_scope="last",
        )
        with controller.apply(model):
            output = model(torch.zeros(1, 1, 2))

        self.assertAlmostEqual(float(output[0, 0, 0]), 4.0)
        self.assertGreater(controller.control_cost(), 0.0)
        diagnostics = controller.diagnostics()
        self.assertEqual(diagnostics["generation_tracking_steps"], 1.0)
        self.assertGreater(diagnostics["generation_tracking_mae"], 0.0)

    def test_adaptive_state_capture_resets_generation_history_before_replay(self):
        model = ReplayModel()
        controller = AdaptiveActivationController(
            "pid",
            0,
            torch.tensor([1.0, 0.0]),
            setpoint=3.0,
            kp=0.0,
            ki=1.0,
            token_scope="last",
        )

        _, _, states, _, execution = run_and_capture(
            model,
            ReplayTokenizer(),
            "question",
            controller,
            [0],
            {"max_new_tokens": 2, "do_sample": False, "num_beams": 1},
        )

        self.assertEqual(execution["generation_tracking_steps"], 2.0)
        self.assertAlmostEqual(float(states[0][0]), 3.0)

    def test_mapped_prompt_intervention_uses_declared_rewrite_and_edit_cost(self):
        class Tokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return text.split()

        intervention = MappedPromptIntervention(
            "paraphrase",
            {"answer clearly": "state the answer clearly"},
        )
        prepared = intervention.prepare_prompt("answer clearly")

        self.assertEqual(prepared, "state the answer clearly")
        self.assertEqual(intervention.control_cost(Tokenizer()), 2.0)

    def test_adaptive_intervention_accepts_setpoint_sweep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            np.save(base_dir / "direction.npy", np.array([1.0, 0.0], dtype=np.float32))
            interventions = build_interventions(
                [
                    {
                        "name": "controller",
                        "type": "adaptive",
                        "layer": 0,
                        "direction": "direction.npy",
                        "setpoints": [-1.0, 0.0, 1.0],
                    }
                ],
                base_dir=base_dir,
            )

        self.assertEqual(len(interventions), 4)
        self.assertEqual(
            [intervention.setpoint for intervention in interventions[1:]],
            [-1.0, 0.0, 1.0],
        )

    def test_reachable_geometry_uses_example_baselines(self):
        samples = [
            make_sample("a", 0, ControlChannel.BASELINE, [10.0, 10.0]),
            make_sample("a", 0, ControlChannel.PROMPT, [11.0, 10.0]),
            make_sample("a", 0, ControlChannel.ACTIVATION, [10.0, 11.0]),
            make_sample("b", 0, ControlChannel.BASELINE, [-10.0, -10.0]),
            make_sample("b", 0, ControlChannel.PROMPT, [-9.0, -10.0]),
            make_sample("b", 0, ControlChannel.ACTIVATION, [-10.0, -9.0]),
        ]

        prompt = baseline_displacements(samples, channel=ControlChannel.PROMPT)
        activation = baseline_displacements(samples, channel=ControlChannel.ACTIVATION)
        rows = summarize_reachability(samples)

        self.assertTrue(np.allclose(prompt, [[1.0, 0.0], [1.0, 0.0]]))
        self.assertEqual(effective_rank(prompt, center=False), 1.0)
        self.assertAlmostEqual(subspace_overlap(prompt, activation), 0.0, places=6)
        self.assertTrue(
            any(row["channel"] == "prompt_activation_overlap" for row in rows)
        )
        self.assertIn("n_components", rows[0])

    def test_geometry_reports_target_budget_and_layer_propagation(self):
        samples = [
            make_sample(
                "a",
                0,
                ControlChannel.BASELINE,
                [0.0, 0.0],
                target_projection=0.0,
            ),
            make_sample(
                "a",
                1,
                ControlChannel.BASELINE,
                [0.0, 0.0],
                target_projection=0.0,
            ),
            make_sample(
                "a",
                0,
                ControlChannel.ACTIVATION,
                [1.0, 2.0],
                name="activation_addition_a1",
                target_projection=1.0,
                parameters={"strength": 1.0},
            ),
            make_sample(
                "a",
                1,
                ControlChannel.ACTIVATION,
                [2.0, 4.0],
                name="activation_addition_a1",
                target_projection=2.0,
                parameters={"strength": 1.0},
            ),
        ]

        target_rows = target_orthogonal_decomposition(samples)
        growth_rows = budget_growth(samples)
        propagation = layerwise_propagation(samples)

        self.assertAlmostEqual(
            target_rows[0]["orthogonal_displacement"],
            2.0,
        )
        self.assertEqual(growth_rows[0]["n_examples"], 1)
        self.assertAlmostEqual(propagation[0]["expansion_ratio"], 2.0)

    def test_controllability_boundary_brackets_first_failed_control(self):
        samples = [
            make_sample(
                "a",
                0,
                ControlChannel.BASELINE,
                [0.0, 0.0],
                target_projection=0.0,
            ),
            make_sample(
                "a",
                0,
                ControlChannel.ACTIVATION,
                [1.0, 0.0],
                name="activation_addition_a1",
                target_projection=1.0,
                parameters={"strength": 1.0},
            ),
            make_sample(
                "a",
                0,
                ControlChannel.ACTIVATION,
                [2.0, 0.0],
                name="activation_addition_a2",
                target_projection=2.0,
                parameters={"strength": 2.0},
            ),
            make_sample(
                "a",
                0,
                ControlChannel.ACTIVATION,
                [4.0, 0.0],
                name="activation_addition_a4",
                preserved=False,
                target_projection=4.0,
                parameters={"strength": 4.0},
                constraint_results={"task": False, "control_budget": True},
            ),
        ]

        rows = controllability_boundary_rows(samples)
        summaries = summarize_controllability_boundaries(
            rows,
            bootstrap_resamples=20,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["boundary_status"], "bracketed")
        self.assertEqual(rows[0]["lower_control_bound"], 2.0)
        self.assertEqual(rows[0]["upper_control_bound"], 4.0)
        self.assertEqual(rows[0]["binding_constraints"], "task")
        self.assertEqual(summaries[0]["bracketed_rate"], 1.0)

    def test_prompt_boundary_is_labeled_sample_limited(self):
        samples = [
            make_sample(
                "a",
                0,
                ControlChannel.BASELINE,
                [0.0],
                target_projection=0.0,
            ),
            make_sample(
                "a",
                0,
                ControlChannel.PROMPT,
                [1.0],
                name="optimized_prompt_0000",
                target_projection=1.0,
                control_cost=8.0,
            ),
        ]

        rows = controllability_boundary_rows(samples)

        self.assertEqual(rows[0]["ordered_control"], 0)
        self.assertEqual(rows[0]["boundary_status"], "sample_limited")

    def test_directed_accessibility_detects_channel_mismatch(self):
        aligned = [
            make_sample(
                "aligned",
                0,
                ControlChannel.BASELINE,
                [0.0, 0.0],
            ),
            make_sample("aligned", 0, ControlChannel.PROMPT, [1.0, 0.0], name="p1"),
            make_sample("aligned", 0, ControlChannel.PROMPT, [2.0, 0.0], name="p2"),
            make_sample("aligned", 0, ControlChannel.ACTIVATION, [1.0, 0.0], name="a1"),
            make_sample("aligned", 0, ControlChannel.ACTIVATION, [2.0, 0.0], name="a2"),
        ]
        mismatched = [
            make_sample(
                "mismatched",
                0,
                ControlChannel.BASELINE,
                [0.0, 0.0],
            ),
            make_sample("mismatched", 0, ControlChannel.PROMPT, [1.0, 0.0], name="p1"),
            make_sample("mismatched", 0, ControlChannel.PROMPT, [2.0, 0.0], name="p2"),
            make_sample(
                "mismatched", 0, ControlChannel.ACTIVATION, [0.0, 1.0], name="a1"
            ),
            make_sample(
                "mismatched", 0, ControlChannel.ACTIVATION, [0.0, 2.0], name="a2"
            ),
        ]

        rows = directed_accessibility_rows(aligned + mismatched, subsample_repeats=8)
        by_example = {(row["example_id"], row["direction"]): row for row in rows}
        summaries = summarize_directed_accessibility(
            rows,
            bootstrap_resamples=20,
        )

        self.assertEqual(
            by_example[("aligned", "prompt_to_activation")]["mean_normalized_gap"],
            0.0,
        )
        self.assertGreater(
            by_example[("mismatched", "prompt_to_activation")]["mean_normalized_gap"],
            0.5,
        )
        self.assertEqual(len(summaries), 2)

    def test_detection_control_gap_separates_probe_signal_from_movement(self):
        samples = [
            make_sample(
                "negative",
                0,
                ControlChannel.BASELINE,
                [0.0],
                label=0.0,
                target_projection=0.0,
            ),
            make_sample(
                "positive",
                0,
                ControlChannel.BASELINE,
                [2.0],
                label=1.0,
                target_projection=2.0,
            ),
            make_sample(
                "negative",
                0,
                ControlChannel.PROMPT,
                [0.1],
                target_projection=0.1,
            ),
            make_sample(
                "positive",
                0,
                ControlChannel.PROMPT,
                [2.1],
                preserved=False,
                target_projection=2.1,
            ),
        ]

        rows = detection_control_gap_rows(samples)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["natural_oriented_projection_auc"], 1.0)
        self.assertAlmostEqual(rows[0]["control_to_detection_ratio"], 0.025)
        self.assertEqual(rows[0]["controlled_example_rate"], 0.5)
        self.assertEqual(rows[0]["low_control_fraction"], 1.0)

    def test_directed_accessibility_keeps_empty_target_failures(self):
        samples = [
            make_sample("a", 0, ControlChannel.BASELINE, [0.0, 0.0]),
            make_sample("a", 0, ControlChannel.PROMPT, [1.0, 0.0]),
            make_sample(
                "a",
                0,
                ControlChannel.ACTIVATION,
                [0.0, 1.0],
                preserved=False,
            ),
        ]

        rows = directed_accessibility_rows(samples)
        prompt_to_activation = next(
            row for row in rows if row["direction"] == "prompt_to_activation"
        )
        summary = summarize_directed_accessibility(
            rows,
            bootstrap_resamples=20,
        )

        self.assertEqual(prompt_to_activation["status"], "target_empty")
        self.assertEqual(prompt_to_activation["coverage_rate"], 0.0)
        self.assertEqual(
            next(row for row in summary if row["direction"] == "prompt_to_activation")[
                "estimable_example_rate"
            ],
            0.0,
        )

    def test_trajectory_summary_uses_ordered_control_values(self):
        samples = [
            make_sample(
                "a",
                0,
                ControlChannel.ACTIVATION,
                [strength, strength**2],
                name=f"activation_addition_a{strength}",
                parameters={"strength": strength},
            )
            for strength in (-1.0, 0.0, 1.0)
        ]

        rows = summarize_trajectories(samples)

        self.assertEqual(rows[0]["n_points"], 3)
        self.assertGreater(rows[0]["mean_curvature"], 0.0)

    def test_principal_angles_identify_matching_subspaces(self):
        first = np.array([[1.0, 0.0], [2.0, 0.0]])
        second = np.array([[3.0, 0.0], [-1.0, 0.0]])
        angles = principal_angles(first, second)
        self.assertTrue(np.allclose(angles, 0.0))

    def test_split_half_stability_uses_disjoint_example_groups(self):
        samples = []
        for index in range(4):
            example_id = str(index)
            samples.extend(
                [
                    make_sample(
                        example_id,
                        0,
                        ControlChannel.BASELINE,
                        [0.0, 0.0],
                    ),
                    make_sample(
                        example_id,
                        0,
                        ControlChannel.PROMPT,
                        [float(index + 1), 0.0],
                    ),
                ]
            )

        rows = split_half_stability_rows(samples, repeats=5, seed=0)
        summary = summarize_split_half_stability(rows)

        self.assertEqual(len(rows), 5)
        self.assertEqual(summary[0]["mean_subspace_overlap"], 1.0)
        self.assertEqual(summary[0]["n_examples_per_half"], 2)

    def test_local_jacobian_rank_recovers_independent_controls(self):
        jacobian = control_jacobian(
            lambda control: torch.stack(
                [control[0] + control[1], control[0] - control[1], control[2]]
            ),
            torch.zeros(3),
        )
        summary = local_controllability(jacobian)

        self.assertEqual(summary["rank"], 3)
        self.assertGreater(summary["minimum_nonzero_gain"], 0.0)

    def test_finite_difference_model_jacobians_follow_control_basis(self):
        class JacobianModel(DummyModel):
            def forward(self, hidden, use_cache=False):
                return super().forward(hidden)

        model = JacobianModel()
        jacobians = finite_difference_jacobians(
            model,
            {"hidden": torch.zeros(1, 1, 2)},
            injection_layer=0,
            capture_layers=[0, 1],
            basis=torch.eye(2),
            epsilon=0.25,
        )

        self.assertTrue(np.allclose(jacobians[0], np.eye(2)))
        self.assertTrue(np.allclose(jacobians[1], np.eye(2)))

    def test_state_archive_round_trip(self):
        samples = [
            make_sample("a", 0, ControlChannel.BASELINE, [0.0, 0.0]),
            make_sample("a", 0, ControlChannel.PROMPT, [1.0, 0.0]),
        ]
        samples[0].token_states = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)
        samples[1].token_states = np.array([[1.0, 0.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            save_state_samples(samples, tmp)
            loaded = load_state_samples(tmp)

        self.assertEqual(len(loaded), 2)
        self.assertTrue(np.array_equal(loaded[1].state, samples[1].state))
        self.assertTrue(np.array_equal(loaded[0].token_states, samples[0].token_states))
        self.assertEqual(loaded[1].intervention.channel, ControlChannel.PROMPT)

    def test_patching_and_cache_restore_hooks(self):
        model = DummyModel()
        hidden = torch.zeros(1, 1, 2)
        with ActivationCache([0]).capture(model) as cache:
            model(hidden)
        source = cache.values[0] + 4.0
        with StatePatching(0, source).apply(model):
            output = model(hidden)

        self.assertTrue(torch.allclose(output, torch.tensor([[[6.0, 6.0]]])))
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)

    def test_attention_head_ablation_zeros_only_selected_head_slice(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.head_dim = 1
                self.o_proj = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.o_proj.weight.copy_(torch.eye(2))

            def forward(self, hidden):
                return self.o_proj(hidden)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()

            def forward(self, hidden):
                return self.self_attn(hidden)

        block = Block()
        hidden = torch.tensor([[[2.0, 3.0]]])
        with AttentionHeadAblation(block, head=0).apply():
            output = block(hidden)

        self.assertTrue(torch.equal(output, torch.tensor([[[0.0, 3.0]]])))

    def test_monitor_invariance_reports_worst_case_failure(self):
        samples = [
            make_sample("a", 0, ControlChannel.BASELINE, [0.0]),
            make_sample("a", 0, ControlChannel.PROMPT, [1.0]),
            make_sample("a", 0, ControlChannel.PROMPT, [2.0], name="prompt_2"),
        ]
        rows = monitor_invariance(samples, [0.9, 0.8, 0.1])
        self.assertEqual(rows[0]["agreement"], 0.5)
        self.assertEqual(rows[0]["worst_case_consistency"], 0.0)

    def test_monitor_split_keeps_question_pairs_together(self):
        samples = []
        for pair in range(4):
            for label, suffix in ((0.0, "neutral"), (1.0, "evaluation")):
                samples.append(
                    make_sample(
                        f"pair_{pair}_{suffix}",
                        0,
                        ControlChannel.BASELINE,
                        [label],
                        label=label,
                        tags={"pair_id": f"pair_{pair}"},
                    )
                )

        train, test = _split_ids(samples, "monitor_label", 0.5, seed=0)

        for pair in range(4):
            ids = {f"pair_{pair}_neutral", f"pair_{pair}_evaluation"}
            self.assertTrue(ids <= train or ids <= test)

    def test_monitor_partition_honors_declared_three_way_split(self):
        samples = []
        expected = {"train": set(), "validation": set(), "test": set()}
        for split, expected_ids in expected.items():
            for label, suffix in ((0.0, "neutral"), (1.0, "evaluation")):
                example_id = f"{split}_{suffix}"
                expected_ids.add(example_id)
                samples.append(
                    make_sample(
                        example_id,
                        0,
                        ControlChannel.BASELINE,
                        [label],
                        label=label,
                        tags={"pair_id": split, "split": split},
                    )
                )

        train, validation, test = _partition_ids(
            samples,
            "monitor_label",
            0.6,
            0.2,
            seed=0,
        )

        self.assertEqual(train, expected["train"])
        self.assertEqual(validation, expected["validation"])
        self.assertEqual(test, expected["test"])

    def test_multiple_comparison_adjustments_are_monotone(self):
        pvalues = [0.01, 0.02, 0.5]
        holm = adjust_pvalues(pvalues, method="holm")
        fdr = adjust_pvalues(pvalues, method="benjamini-hochberg")

        self.assertTrue(np.all(holm >= pvalues))
        self.assertTrue(np.all(fdr >= pvalues))

    def test_monitor_comparison_pairs_training_modes_by_example(self):
        rows = []
        for example_id in ("a", "b"):
            for training in ("natural", "reachable_augmented"):
                rows.append(
                    {
                        "model_name": "dummy",
                        "layer": 0,
                        "monitor": "linear",
                        "training": training,
                        "example_id": example_id,
                        "channel": "baseline",
                        "score": 0.9,
                        "label": 1,
                    }
                )
                rows.append(
                    {
                        "model_name": "dummy",
                        "layer": 0,
                        "monitor": "linear",
                        "training": training,
                        "example_id": example_id,
                        "channel": "prompt",
                        "score": 0.9 if training == "reachable_augmented" else 0.1,
                        "label": 1,
                    }
                )

        comparisons = paired_monitor_comparisons(
            rows,
            seed=0,
            bootstrap_resamples=20,
            permutations=100,
        )
        consistency = next(
            row for row in comparisons if row["metric"] == "worst_case_consistency"
        )

        self.assertEqual(consistency["augmented_minus_natural"], 1.0)

    def test_monitor_models_and_pooling(self):
        states = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        monitor = LinearMonitor().fit(states, labels)
        predictions = monitor.predict(states)
        self.assertTrue(np.array_equal(predictions, labels.astype(bool)))

        ood = MahalanobisOODMonitor().fit(np.array([[0.0], [0.1], [-0.1]]))
        self.assertGreater(
            ood.score(np.array([[3.0]]))[0], ood.score(np.array([[0.0]]))[0]
        )

        hidden = np.array([[[1.0], [2.0]], [[3.0], [100.0]]])
        mask = np.array([[1, 1], [1, 0]])
        self.assertTrue(
            np.array_equal(
                pool_hidden_states(hidden, mask=mask, pooling="last"), [[2.0], [3.0]]
            )
        )

    def test_trajectory_similarity_distinguishes_routes(self):
        same = trajectory_similarity([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
        different = trajectory_similarity(
            [[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]
        )
        self.assertAlmostEqual(same["route_cosine"], 1.0)
        self.assertAlmostEqual(different["route_cosine"], 0.0)

    def test_transfer_study_pairs_tagged_groups(self):
        samples = []
        for example_id, source in (("a", "source_a"), ("b", "source_b")):
            samples.extend(
                [
                    make_sample(
                        example_id,
                        0,
                        ControlChannel.BASELINE,
                        [0.0, 0.0],
                        tags={"source": source},
                        target_projection=0.0,
                    ),
                    make_sample(
                        example_id,
                        0,
                        ControlChannel.PROMPT,
                        [1.0, 0.0],
                        name="prompt_1",
                        tags={"source": source},
                        target_projection=1.0,
                    ),
                    make_sample(
                        example_id,
                        0,
                        ControlChannel.PROMPT,
                        [2.0, 0.0],
                        name="prompt_2",
                        tags={"source": source},
                        target_projection=2.0,
                    ),
                ]
            )

        rows = transfer_rows(samples)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["subspace_overlap"], 1.0)
        self.assertAlmostEqual(rows[0]["transfer_score"], 1.0)

    def test_causal_pair_selection_matches_target_projection(self):
        samples = [
            make_sample(
                "a",
                2,
                ControlChannel.PROMPT,
                [0.0],
                name="optimized_prompt_0000",
                target_projection=-2.0,
            ),
            make_sample(
                "a",
                2,
                ControlChannel.PROMPT,
                [0.0],
                name="optimized_prompt_0001",
                target_projection=1.0,
            ),
            make_sample(
                "a",
                2,
                ControlChannel.ACTIVATION,
                [0.0],
                name="activation_addition_a-1",
                target_projection=-1.9,
            ),
        ]

        pairs = matched_control_pairs(
            samples,
            selection_layer=2,
            prompt_prefix="optimized_prompt",
            activation_prefix="activation_addition",
            target_metric="target_projection",
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].intervention.name, "optimized_prompt_0000")

    def test_cross_prompt_pairs_match_neutral_and_evaluation_records(self):
        samples = [
            make_sample(
                f"pair_{label}",
                2,
                ControlChannel.BASELINE,
                [float(label)],
                label=float(label),
                tags={"pair_id": "pair"},
            )
            for label in (0, 1)
        ]

        pairs = matched_prompt_pairs(samples, selection_layer=2)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], "pair")
        self.assertEqual(pairs[0][1].metrics["monitor_label"], 0.0)
        self.assertEqual(pairs[0][2].metrics["monitor_label"], 1.0)

    def test_control_cost_keeps_unreached_setpoints_as_failures(self):
        samples = [
            make_sample(
                "a",
                0,
                ControlChannel.ACTIVATION,
                [0.0],
                name="activation_addition_a1",
                target_projection=1.05,
            ),
            make_sample(
                "b",
                0,
                ControlChannel.ACTIVATION,
                [0.0],
                name="activation_addition_a1",
                target_projection=3.0,
            ),
        ]

        rows = control_cost_rows(
            samples,
            target_metric="target_projection",
            setpoints=[1.0],
            tolerance=0.1,
        )
        summary = summarize_control_costs(rows)

        self.assertEqual(sum(row["reached"] for row in rows), 1)
        self.assertEqual(summary[0]["n_reached"], 1)
        self.assertEqual(summary[0]["reach_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
