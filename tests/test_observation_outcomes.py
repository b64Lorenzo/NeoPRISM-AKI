"""Observed laboratory values must not generate inferred outcome classifications."""

import csv
import tempfile
import unittest
from pathlib import Path

import tests.test_patient_vectors as patient_fixtures
from src.logging.logger import Logger
from src.preprocessing.parameters.parameter_map import load_parameter_map
from src.preprocessing.patient_pipeline import analyze_inputs
from src.preprocessing.postoperative.processor import PostoperativeProcessor
from src.preprocessing.preoperative.processor import PreoperativeProcessor

ROOT = Path(__file__).resolve().parents[1]


class ObservationOutcomeTests(unittest.TestCase):
    def test_configurations_contain_no_removed_rules_or_derived_output(self):
        forbidden = {
            "baseline_lookback_hours",
            "ratio_stages",
            "absolute_criteria",
            "acute_increase",
            "pairwise_increase",
            "default_stage",
            "creatinine_field",
            "baseline_creatinine_parameter_id",
            "postoperative_window_hours",
            "clinical_reference",
        }
        for name in ("pipeline.xml", "pipeline.template.xml"):
            with self.subTest(name=name):
                mapping = load_parameter_map(ROOT / "config" / name)
                self.assertTrue(forbidden.isdisjoint(mapping["patient_outcome_rules"]))
                self.assertNotIn("aki_kdigo_stage", {item["id"] for item in mapping["parameters"]})
                self.assertNotIn(
                    "aki_kdigo_stage",
                    {item["id"] for item in mapping["postoperative_measurements"]},
                )

    def test_low_and_high_creatinine_without_outcome_evidence_stay_unclassified(self):
        mapping = load_parameter_map(ROOT / "config" / "pipeline.template.xml")
        with tempfile.TemporaryDirectory() as temporary:
            for value in ("45", "400"):
                with self.subTest(value=value):
                    source = patient_fixtures.PatientVectorTests().make_patient(
                        Path(temporary),
                        "synthetic.csv",
                        postop_creatinine=value,
                        deceased="",
                    )
                    pre = PreoperativeProcessor(mapping).process(source)
                    result = PostoperativeProcessor(mapping).process(source, pre)
                    self.assertEqual(pre.features["baseline_creatinine"], "40")
                    self.assertEqual(result.measurements["postop_creatinine_72h"], value)
                    self.assertIsNone(result.target)
                    self.assertEqual(result.target_status, "unclassified")

    def test_explicit_outcome_categories_are_not_overridden_by_lab_values(self):
        mapping = load_parameter_map(ROOT / "config" / "pipeline.template.xml")
        with tempfile.TemporaryDirectory() as temporary:
            for category, expected in (("optimal", 1), ("severe problems", -1)):
                with self.subTest(category=category):
                    source = patient_fixtures.PatientVectorTests().make_patient(
                        Path(temporary),
                        "synthetic.csv",
                        postop_creatinine="400",
                        deceased="",
                        operation_result=category,
                    )
                    pre = PreoperativeProcessor(mapping).process(source)
                    self.assertEqual(
                        PostoperativeProcessor(mapping).process(source, pre).target, expected
                    )

    def test_csv_export_keeps_observations_without_the_removed_column(self):
        mapping = load_parameter_map(ROOT / "config" / "pipeline.template.xml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = patient_fixtures.PatientVectorTests().make_patient(
                root, "synthetic.csv", postop_creatinine="400", deceased=""
            )
            with Logger(log_root=root / "logs") as logger:
                analyze_inputs([source], root / "results", mapping, logger, generate_plots=False)
            with (root / "results/postoperative_measurements.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                reader = csv.DictReader(stream)
                self.assertNotIn("aki_kdigo_stage", reader.fieldnames)
                self.assertEqual(next(reader)["postop_creatinine_72h"], "400")
            with (root / "results/patient_outcomes.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(next(csv.DictReader(stream))["target"], "NA")
