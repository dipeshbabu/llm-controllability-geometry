import json
import tempfile
import unittest
from pathlib import Path

from llm_controllability.cli import _load_texts, build_parser
from llm_controllability.optimization.robustness import deterministic_variants
from llm_controllability.orchestration import load_matrix, matrix_commands
from llm_controllability.targets.specs import token_id_from_text


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == " one":
            return [1]
        return [1, 2]


class SpecAndCliTests(unittest.TestCase):
    def test_compute_commands_default_to_automatic_device_selection(self):
        parser = build_parser()

        run = parser.parse_args(["run", "--spec", "spec.json", "--out", "runs/x"])
        scope = parser.parse_args(
            [
                "gemma-scope",
                "--states-dir",
                "runs/x",
                "--out-dir",
                "runs/scope",
                "--model-name",
                "google/gemma-3-4b-it",
                "--layer",
                "12",
            ]
        )
        info = parser.parse_args(["runtime-info"])

        self.assertEqual(run.device_map, "auto")
        self.assertEqual(scope.device, "auto")
        self.assertEqual(info.device_map, "auto")

    def test_token_id_from_text_requires_single_token(self):
        self.assertEqual(token_id_from_text(DummyTokenizer(), " one"), 1)
        with self.assertRaises(ValueError):
            token_id_from_text(DummyTokenizer(), "two tokens")

    def test_deterministic_variants_are_named_and_unique(self):
        variants = deterministic_variants(" Hello   world ")
        names = [name for name, _ in variants]

        self.assertIn("original", names)
        self.assertIn("instruction_wrap", names)
        self.assertEqual(len({text for _, text in variants}), len(variants))

    def test_cli_parser_accepts_run_command(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--spec",
                "spec.json",
                "--out",
                "runs/x",
                "--methods",
                "random",
                "--contexts",
                "train.jsonl",
                "--context-count",
                "16",
            ]
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.methods, ["random"])
        self.assertEqual(args.context_count, 16)

    def test_jsonl_text_loader_reads_prompt_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "contexts.jsonl"
            path.write_text(
                json.dumps({"prompt": "first"}) + "\n"
                + json.dumps({"prompt": "second"}) + "\n",
                encoding="utf-8",
            )
            texts = _load_texts(path)

        self.assertEqual(texts, ["first", "second"])

    def test_cli_parser_accepts_frontier_data_command(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "build-frontier-data",
                "--out-dir",
                "data/frontier",
                "--sources",
                "mmlu_pro",
                "math500",
            ]
        )

        self.assertEqual(args.command, "build-frontier-data")
        self.assertEqual(args.sources, ["mmlu_pro", "math500"])

    def test_cli_parser_accepts_controllability_commands(self):
        parser = build_parser()
        collect = parser.parse_args(
            ["collect-reachable", "--spec", "study.json", "--out", "runs/study"]
        )
        monitor = parser.parse_args(
            ["monitor-invariance", "--states-dir", "runs/study", "--out-dir", "runs/monitor"]
        )
        causal = parser.parse_args(
            [
                "causal-study",
                "--spec",
                "study.json",
                "--states-dir",
                "runs/study",
                "--out-dir",
                "runs/causal",
            ]
        )
        control = parser.parse_args(
            [
                "analyze-control",
                "--spec",
                "study.json",
                "--states-dir",
                "runs/study",
                "--out-dir",
                "runs/control",
            ]
        )
        jacobians = parser.parse_args(
            [
                "analyze-jacobians",
                "--spec",
                "study.json",
                "--out",
                "runs/jacobians.csv",
            ]
        )

        self.assertEqual(collect.command, "collect-reachable")
        self.assertEqual(monitor.command, "monitor-invariance")
        self.assertEqual(causal.command, "causal-study")
        self.assertEqual(control.command, "analyze-control")
        self.assertEqual(jacobians.command, "analyze-jacobians")

    def test_recommended_matrix_drives_every_launcher(self):
        matrix = load_matrix(
            Path(__file__).parents[1] / "configs" / "recommended_matrix.json"
        )
        controllability = matrix_commands(matrix, "controllability")
        token_monitors = matrix_commands(matrix, "token_monitors")
        gemma_scope = matrix_commands(matrix, "gemma_scope")

        self.assertEqual(len(controllability), 9)
        self.assertEqual(len(token_monitors), 5)
        self.assertEqual(len(gemma_scope), 4)
        self.assertTrue(
            all(command[1] == "scripts/run_model_controllability.sh"
                for command in controllability)
        )
        self.assertTrue(
            all(command[1] == "scripts/run_token_monitor_study.sh"
                for command in token_monitors)
        )
        self.assertTrue(
            all(command[1] == "scripts/run_gemma_scope_study.sh"
                for command in gemma_scope)
        )

    def test_cli_parser_accepts_matrix_launcher(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-study-matrix",
                "--analysis",
                "controllability",
                "--dry-run",
            ]
        )

        self.assertEqual(args.analysis, "controllability")
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()
