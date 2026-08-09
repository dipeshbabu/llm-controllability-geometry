import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from llm_controllability.activations.probes import _to_numpy_float32
from llm_controllability.controllability.specs import build_study_spec
from llm_controllability.controllability.study import _limit_examples
from llm_controllability.data.frontier import (
    FrontierQuestion,
    frontier_question_from_row,
    split_items,
    write_frontier_bundle,
)
from llm_controllability.optimization.robustness import robustness_summary_rows
from llm_controllability.reporting.latex import rows_to_latex_table
from llm_controllability.reporting.results import CandidateRecord
from llm_controllability.targets.directions import (
    mean_difference_direction,
    projection_gap,
    top_direction_specs,
)
from llm_controllability.targets.generation import (
    logit_specs,
    neuron_specs,
    parse_int_list,
    write_spec,
)


class ResearchWorkflowTests(unittest.TestCase):
    def test_example_limit_preserves_pairs_and_all_declared_splits(self):
        examples = []
        for split, pair_count in (
            ("train", 6),
            ("validation", 2),
            ("test", 2),
        ):
            for pair in range(pair_count):
                for variant in ("neutral", "evaluation"):
                    examples.append(
                        {
                            "id": f"{split}_{pair}_{variant}",
                            "pair_id": f"{split}_{pair}",
                            "split": split,
                        }
                    )

        selected = _limit_examples(examples, 10)
        selected_ids = {example["id"] for example in selected}

        self.assertEqual(len(selected), 10)
        self.assertEqual(
            {example["split"] for example in selected},
            {"train", "validation", "test"},
        )
        for example in selected:
            counterpart = (
                example["id"].replace("_neutral", "_evaluation")
                if example["id"].endswith("_neutral")
                else example["id"].replace("_evaluation", "_neutral")
            )
            self.assertIn(counterpart, selected_ids)

    def test_frontier_split_is_stratified_by_source(self):
        items = [
            FrontierQuestion(source=source, question=f"{source}-{index}")
            for source in ("a", "b")
            for index in range(4)
        ]

        train, evaluation = split_items(items, train_fraction=0.5, seed=0)

        self.assertEqual(
            {source: sum(item.source == source for item in train) for source in ("a", "b")},
            {"a": 2, "b": 2},
        )
        self.assertEqual(
            {
                source: sum(item.source == source for item in evaluation)
                for source in ("a", "b")
            },
            {"a": 2, "b": 2},
        )

    def test_parse_int_list_supports_ranges(self):
        self.assertEqual(parse_int_list("1,3-5"), [1, 3, 4, 5])

    def test_target_generation_writes_spec(self):
        targets = logit_specs([" dog"]) + neuron_specs([1], [2])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            write_spec(
                targets,
                path,
                model_size="70m",
                attn_implementation="eager",
                device_map="cuda",
            )
            data = json.loads(path.read_text())

        self.assertEqual(len(data["targets"]), 2)
        self.assertEqual(data["model_size"], "70m")
        self.assertEqual(data["attn_implementation"], "eager")
        self.assertEqual(data["device_map"], "cuda")

    def test_direction_math_and_top_specs(self):
        a = np.array([[2.0, 0.0], [2.0, 1.0]])
        b = np.array([[0.0, 0.0], [0.0, 1.0]])
        direction = mean_difference_direction(a, b)
        self.assertGreater(projection_gap(a, b, direction), 0)
        specs = top_direction_specs(
            [
                {"name": "x", "layer": 0, "projection_gap": 1.0, "vector_path": "x0.npy"},
                {"name": "x", "layer": 1, "projection_gap": 3.0, "vector_path": "x1.npy"},
            ],
            top_k=1,
        )
        self.assertEqual(specs[0]["layer"], 1)
        bidirectional = top_direction_specs(
            [
                {
                    "name": "x",
                    "layer": 1,
                    "projection_gap": 3.0,
                    "vector_path": "x1.npy",
                }
            ],
            top_k=1,
            bidirectional=True,
        )
        self.assertEqual(len(bidirectional), 2)
        self.assertEqual(
            [spec["minimize"] for spec in bidirectional],
            [True, False],
        )

        held_out_specs = top_direction_specs(
            [
                {
                    "name": "x",
                    "layer": 0,
                    "projection_gap": 5.0,
                    "eval_projection_gap": 1.0,
                    "vector_path": "x0.npy",
                },
                {
                    "name": "x",
                    "layer": 1,
                    "projection_gap": 2.0,
                    "eval_projection_gap": 3.0,
                    "vector_path": "x1.npy",
                },
            ],
            top_k=1,
        )
        self.assertEqual(held_out_specs[0]["layer"], 1)

        stable_specs = top_direction_specs(
            [
                {
                    "name": "x",
                    "layer": 0,
                    "projection_gap": 5.0,
                    "eval_projection_gap": -4.0,
                    "vector_path": "x0.npy",
                },
                {
                    "name": "x",
                    "layer": 1,
                    "projection_gap": 2.0,
                    "eval_projection_gap": 1.0,
                    "vector_path": "x1.npy",
                },
            ],
            top_k=1,
        )
        self.assertEqual(stable_specs[0]["layer"], 1)

        with self.assertRaisesRegex(ValueError, "positive held-out"):
            top_direction_specs(
                [
                    {
                        "name": "x",
                        "layer": 0,
                        "projection_gap": 5.0,
                        "eval_projection_gap": -1.0,
                        "vector_path": "x0.npy",
                    }
                ],
                top_k=1,
            )

    def test_bfloat16_states_convert_to_numpy_float32(self):
        tensor = torch.ones((2, 3), dtype=torch.bfloat16)

        array = _to_numpy_float32(tensor)

        self.assertEqual(array.dtype, np.float32)
        self.assertEqual(array.shape, (2, 3))

    def test_study_spec_uses_positive_held_out_direction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.npy"
            second = root / "second.npy"
            np.save(first, np.array([1.0, 0.0], dtype=np.float32))
            np.save(second, np.array([0.0, 1.0], dtype=np.float32))
            sweep = root / "sweep.csv"
            fields = [
                "name",
                "layer",
                "projection_gap",
                "eval_projection_gap",
                "a_mean_projection",
                "b_mean_projection",
                "vector_path",
            ]
            with sweep.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "name": "direction",
                            "layer": 0,
                            "projection_gap": 5.0,
                            "eval_projection_gap": -4.0,
                            "a_mean_projection": 1.0,
                            "b_mean_projection": -1.0,
                            "vector_path": first,
                        },
                        {
                            "name": "direction",
                            "layer": 1,
                            "projection_gap": 2.0,
                            "eval_projection_gap": 1.0,
                            "a_mean_projection": 0.5,
                            "b_mean_projection": -0.5,
                            "vector_path": second,
                        },
                    ]
                )
            data = root / "data.jsonl"
            prompts = root / "prompts.txt"
            natural = root / "natural.txt"
            data.write_text("{}\n", encoding="utf-8")
            prompts.write_text("control\n", encoding="utf-8")
            natural.write_text(
                json.dumps(
                    [
                        {
                            "source_prompt": "question",
                            "controls": {
                                "natural_paraphrase": "rewritten question",
                                "style_concise": "concise question",
                                "capitalization": "QUESTION",
                                "topic_matched": "Topic: question",
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            out = root / "study.json"

            spec = build_study_spec(
                out_path=out,
                model_name="dummy",
                data_path=data,
                direction_sweep=sweep,
                capture_layers=None,
                prompt_controls_path=prompts,
                natural_controls_path=natural,
                cmap_direction_budget=6,
                cmap_query_budget=120,
                cmap_seeds=[0, 1, 2],
            )

            activation = next(
                item
                for item in spec["interventions"]
                if item["name"] == "activation_addition"
            )
            random_control = next(
                item
                for item in spec["interventions"]
                if item["name"] == "orthogonal_random"
            )
            random_path = (out.parent / random_control["direction"]).resolve()
            random_vector = np.load(random_path)
            surface_control = next(
                item
                for item in spec["interventions"]
                if item["name"] == "surface_control"
            )

        self.assertEqual(activation["layer"], 1)
        self.assertEqual(random_control["channel"], "random")
        self.assertEqual(surface_control["type"], "mapped_prompt")
        self.assertEqual(spec["cmap"]["direction_budget"], 6)
        self.assertEqual(spec["cmap"]["query_budget"], 120)
        self.assertEqual(spec["cmap"]["layer"], 1)
        self.assertEqual(spec["cmap"]["seeds"], [0, 1, 2])
        self.assertAlmostEqual(float(random_vector[1]), 0.0, places=6)

    def test_robustness_summary(self):
        rows = robustness_summary_rows(
            [
                CandidateRecord(
                    "t",
                    "epo:lower",
                    0,
                    "x",
                    target=1.2,
                    xentropy=2.0,
                    extra={"base_method": "epo", "variant": "lower", "base_target": 1.0, "base_xentropy": 2.0},
                )
            ],
            target_tolerance=0.25,
        )
        self.assertEqual(rows[0]["survival_rate"], 1.0)

    def test_latex_table_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.tex"
            rows_to_latex_table(
                [{"method": "epo", "best_target": -1.2345}],
                path,
                columns=["method", "best_target"],
            )
            text = path.read_text()

        self.assertIn("\\begin{table}", text)
        self.assertIn("epo", text)

    def test_frontier_data_bundle_writes_expected_files(self):
        item = frontier_question_from_row(
            "mmlu_pro",
            {
                "question": "Which option is correct?",
                "options": ["first", "second"],
                "answer": "A",
                "category": "logic",
            },
        )
        self.assertIsNotNone(item)

        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_frontier_bundle(
                [item],
                tmp,
                train_fraction=1.0,
                validation_fraction=0.0,
            )
            out = Path(tmp)

            self.assertEqual(manifest["n_total"], 1)
            self.assertTrue((out / "text_pools" / "frontier_train.txt").exists())
            self.assertTrue((out / "contrasts" / "eval_awareness_train.json").exists())
            self.assertTrue(
                (out / "contrasts" / "eval_awareness_validation.json").exists()
            )
            self.assertTrue((out / "behavior" / "frontier_behavior.json").exists())
            self.assertTrue(
                (out / "behavior" / "controllability_test.jsonl").exists()
            )
            self.assertTrue((out / "controls" / "prompt_rewrites.json").exists())
            behavior = json.loads(
                (out / "behavior" / "frontier_behavior.json").read_text()
            )

        self.assertIn("continuations", behavior[0])

    def test_gpqa_choices_are_shuffled_with_a_recomputed_answer(self):
        item = frontier_question_from_row(
            "gpqa_diamond",
            {
                "Question": "Which answer is correct?",
                "Correct Answer": "correct",
                "Incorrect Answer 1": "wrong one",
                "Incorrect Answer 2": "wrong two",
                "Incorrect Answer 3": "wrong three",
            },
        )

        self.assertIsNotNone(item)
        assert item is not None
        answer_index = ord(item.answer or "") - ord("A")
        self.assertEqual(item.choices[answer_index], "correct")


if __name__ == "__main__":
    unittest.main()
