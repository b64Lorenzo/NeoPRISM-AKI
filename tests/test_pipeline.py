import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from src.cli import main
from src.logging.logger import Logger
from src.preprocessing.row_pipeline import analyze_inputs, classify_outcome, load_parameter_map

ROOT = Path(__file__).resolve().parents[1]
PARAMETER_MAP = ROOT / "config" / "pipeline.xml"


class PipelineTests(unittest.TestCase):
    def make_input(self, root: Path) -> Path:
        source = root / "sample.csv"
        with source.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "epic.DEMOGRAPHICS.age",
                    "epic.DEMOGRAPHICS.sex",
                    "epic.DEMOGRAPHICS.gewicht",
                    "q_umcg_aki_marco.adm_ventilationdays",
                    "operation result",
                    "epic.AN-15_intellivu.ABP mean [1120000376] (mmHg)",
                ]
            )
            writer.writerow(["10", "F", "20", "1.0", "mid condition", "55"])
            writer.writerow(["", "M", "", "3.5", "severe problems", "60"])
            writer.writerow(["NA", "", "0", "", "optimal condition", "65"])
        return source

    def test_cli_failure_logs_and_prints_same_detailed_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            errors = io.StringIO()
            with patch("src.cli.ROOT", root), patch(
                "src.cli.run_pipeline", side_effect=RuntimeError("bad config")
            ), redirect_stderr(errors):
                exit_code = main(["run", "--config", str(root / "broken.xml")])

            expected = "Command failed (RuntimeError): bad config"
            self.assertEqual(exit_code, 2)
            self.assertEqual(errors.getvalue().splitlines()[-1], expected)
            log_files = list((root / "logs").glob("*/run.log"))
            self.assertEqual(len(log_files), 1)
            self.assertIn(expected, log_files[0].read_text(encoding="utf-8"))

    def test_extracts_only_requested_phases_and_labels_postoperative_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_input(root)
            logs = io.StringIO()
            with redirect_stderr(logs), Logger(log_root=root / "logs") as logger:
                analyze_inputs(
                    [source],
                    root / "results",
                    PARAMETER_MAP,
                    logger,
                    outcome_column="operation result",
                )

            preop = root / "results/extracted/preoperative/input_001.csv"
            postop = root / "results/extracted/postoperative/input_001.csv"
            with preop.open(newline="", encoding="utf-8") as stream:
                preop_rows = list(csv.DictReader(stream))
            with postop.open(newline="", encoding="utf-8") as stream:
                postop_rows = list(csv.DictReader(stream))

            self.assertEqual(len(preop_rows), 3)
            self.assertEqual(
                list(preop_rows[0]),
                [
                    "source_record",
                    "epic.DEMOGRAPHICS.age",
                    "epic.DEMOGRAPHICS.sex",
                    "epic.DEMOGRAPHICS.gewicht",
                ],
            )
            self.assertEqual(postop_rows[0]["outcome_label"], "success")
            self.assertEqual(
                [row["outcome_label"] for row in postop_rows],
                ["success", "failure", "success"],
            )
            self.assertNotIn(
                "epic.AN-15_intellivu.ABP mean [1120000376] (mmHg)",
                preop_rows[0],
            )
            self.assertNotIn(
                "epic.AN-15_intellivu.ABP mean [1120000376] (mmHg)",
                postop_rows[0],
            )
            self.assertIn("analysis completed", logs.getvalue())
            self.assertNotIn("sample.csv", logs.getvalue())
            report = (root / "results/missingness_report.md").read_text(encoding="utf-8")
            self.assertNotIn("sample.csv", report)

    def test_preoperative_missingness_study_and_heatmap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_input(root)
            with redirect_stderr(io.StringIO()), Logger(log_root=root / "logs") as logger:
                analyze_inputs(
                    [source],
                    root / "results",
                    PARAMETER_MAP,
                    logger,
                    outcome_column="operation result",
                )

            summary_path = root / "results/missingness_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as stream:
                summary = list(csv.DictReader(stream))
            age = next(row for row in summary if row["source_column"] == "epic.DEMOGRAPHICS.age")
            weight = next(row for row in summary if row["source_column"] == "epic.DEMOGRAPHICS.gewicht")
            height = next(row for row in summary if row["parameter"] == "Height")
            self.assertEqual(age["missing_count"], "2")
            self.assertEqual(weight["missing_count"], "1")
            self.assertEqual(height["status"], "column_missing")
            self.assertNotIn("severity", age)
            self.assertNotIn("severity", weight)

            with (root / "results/missingness_by_record.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["missing_count"], "0")
            self.assertEqual(records[1]["missing_count"], "2")
            self.assertEqual(records[2]["missing_count"], "2")

            self.assertTrue((root / "results/missingness_patterns.csv").is_file())
            self.assertTrue((root / "results/pairwise_missingness.csv").is_file())
            self.assertTrue((root / "results/missingness_heatmaps/input_001.html").is_file())
            report = (root / "results/missingness_report.md").read_text(encoding="utf-8")
            self.assertIn("complete cases", report.lower())
            self.assertIn("do not determine the missingness mechanism", report)

    def test_no_matched_preoperative_fields_do_not_count_as_complete_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "opaque.csv"
            source.write_text("unmapped_field\n1\n2\n", encoding="utf-8")
            with redirect_stderr(io.StringIO()), Logger(log_root=root / "logs") as logger:
                reports = analyze_inputs(
                    [source], root / "results", PARAMETER_MAP, logger
                )
            self.assertEqual(reports[0]["preoperative_fields"], 0)
            self.assertEqual(reports[0]["preoperative_complete_records"], 0)
            self.assertEqual(reports[0]["preoperative_complete_percent"], 0.0)
            with (root / "results/postoperative_labels.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                labels = list(csv.DictReader(stream))
            self.assertEqual([row["label"] for row in labels], ["unclassified", "unclassified"])

    def test_present_all_missing_column_is_distinct_from_absent_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "missing.csv"
            source.write_text(
                "epic.DEMOGRAPHICS.age;epic.DEMOGRAPHICS.sex\nNA;\nN/A;M\nnull;F\n",
                encoding="utf-8-sig",
            )
            with redirect_stderr(io.StringIO()), Logger(log_root=root / "logs") as logger:
                analyze_inputs([source], root / "results", PARAMETER_MAP, logger)
            with (root / "results/missingness_summary.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            age = next(row for row in rows if row["source_column"] == "epic.DEMOGRAPHICS.age")
            height = next(row for row in rows if row["parameter"] == "Height")
            self.assertEqual(age["status"], "present")
            self.assertEqual(age["missing_count"], "3")
            self.assertNotIn("severity", age)
            self.assertEqual(height["status"], "column_missing")

    def test_only_explicit_outcome_values_receive_a_class(self):
        rules = load_parameter_map(PARAMETER_MAP)["patient_outcome_rules"]["categorical_rules"]
        success_values = tuple(
            value for rule in rules if rule["classification"] == "success" for value in rule["values"]
        )
        failure_values = tuple(
            value for rule in rules if rule["classification"] == "failure" for value in rule["values"]
        )
        def classify(value):
            return classify_outcome(value, success_values, failure_values)
        self.assertEqual(classify("  MID CONDITION "), "success")
        self.assertEqual(classify("optimal condition"), "success")
        self.assertEqual(classify("severe problems"), "failure")
        self.assertEqual(classify("sever problems"), "failure")
        self.assertEqual(classify("death"), "failure")
        self.assertEqual(classify("mild renal concern"), "unclassified")
        self.assertEqual(classify(""), "unclassified")


if __name__ == "__main__":
    unittest.main()
