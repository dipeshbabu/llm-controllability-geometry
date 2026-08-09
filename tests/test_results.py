import json
import tempfile
import unittest
from pathlib import Path

from llm_controllability.controllability.specs import export_prompt_controls
from llm_controllability.reporting.results import (
    CandidateRecord,
    best_target_at_fluent,
    pareto_frontier,
    records_from_csv,
    records_to_csv,
    summarize_by_method,
)


class ResultTests(unittest.TestCase):
    def test_pareto_frontier_for_minimization(self):
        records = [
            CandidateRecord("t", "m", 0, "a", target=3.0, xentropy=1.0),
            CandidateRecord("t", "m", 0, "b", target=2.0, xentropy=2.0),
            CandidateRecord("t", "m", 0, "c", target=4.0, xentropy=3.0),
        ]

        front = pareto_frontier(records, minimize=True)

        self.assertEqual([r.text for r in front], ["a", "b"])

    def test_best_target_at_fluent(self):
        records = [
            CandidateRecord("t", "m", 0, "a", target=5.0, xentropy=1.0),
            CandidateRecord("t", "m", 0, "b", target=1.0, xentropy=9.0),
            CandidateRecord("t", "m", 0, "c", target=3.0, xentropy=2.0),
        ]

        best = best_target_at_fluent(records, quantile=0.67)

        self.assertEqual(best.text, "c")

    def test_csv_roundtrip(self):
        records = [
            CandidateRecord("t", "m", 0, "a", target=1.0, xentropy=2.0, extra={"k": "v"})
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.csv"
            records_to_csv(records, path)
            loaded = records_from_csv(path)

        self.assertEqual(loaded[0].extra, {"k": "v"})
        self.assertEqual(loaded[0].target, 1.0)

    def test_prompt_export_balances_search_seeds_and_writes_provenance(self):
        records = [
            CandidateRecord("t", "epo", seed, f"control {seed}", 1.0 + seed, 1.0)
            for seed in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "records.csv"
            out_path = Path(tmp) / "controls.txt"
            records_to_csv(records, records_path)

            controls = export_prompt_controls(records_path, out_path, top_n=3)
            provenance = json.loads(
                (Path(tmp) / "controls.txt.provenance.json").read_text()
            )

        self.assertEqual(len(controls), 3)
        self.assertEqual({row["seed"] for row in provenance}, {0, 1, 2})

    def test_summarize_by_method(self):
        rows = summarize_by_method(
            [
                CandidateRecord("t", "a", 0, "x", target=2.0, xentropy=1.0),
                CandidateRecord("t", "a", 0, "y", target=1.0, xentropy=3.0),
            ]
        )

        self.assertEqual(rows[0]["best_target"], 1.0)
        self.assertEqual(rows[0]["n"], 2)


if __name__ == "__main__":
    unittest.main()
