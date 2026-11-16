"""Synthetic checks for XML-configurable operation-phase boundaries."""

import csv
import io
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr
from pathlib import Path

from src.logging.logger import Logger
from src.preprocessing.parameters.parameter_map import load_parameter_map, write_parameter_map
from src.preprocessing.patient_pipeline import analyze_inputs
from src.preprocessing.postoperative.processor import PostoperativeProcessor
from src.preprocessing.preoperative.processor import PreoperativeProcessor

ROOT = Path(__file__).resolve().parents[1]


class OperationBoundaryTests(unittest.TestCase):
    def test_configured_postoperative_end_marks_the_section_end(self):
        mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "synthetic.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["unixtime", "epic.EVENTS.EVENT"])
                writer.writerows(
                    [
                        [100, "Start surgery"],
                        [200, "Stop anesthesie"],
                        [250, "Hospital discharge"],
                        [300, "Hospital discharge"],
                    ]
                )
            result = PreoperativeProcessor(mapping).process(source)
        self.assertEqual(result.cutoff, 100)
        self.assertEqual(result.end_cutoff, 200)
        self.assertEqual(result.postoperative_end_cutoff, 250)

    def test_each_default_end_phrase_marks_the_postoperative_start(self):
        mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "synthetic.csv"
            for phrase in mapping["phase_boundaries"]["operation_end"]["phrases"]:
                with self.subTest(phrase=phrase):
                    with source.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(["unixtime", "epic.EVENTS.EVENT"])
                        writer.writerows([[100, "Start surgery"], [200, f"{phrase} | completed"]])
                    result = PreoperativeProcessor(mapping).process(source)
                    self.assertEqual((result.cutoff, result.end_cutoff), (100, 200))

    def test_custom_event_column_and_phrases_split_pre_during_post(self):
        mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
        mapping["phase_boundaries"] = {
            "operation_start": {"event_fields": ["custom.phase"], "phrases": ["BEGIN-X"]},
            "operation_end": {"event_fields": ["custom.phase"], "phrases": ["FINISH-Y"]},
            "postoperative_end": {"event_fields": ["custom.phase"], "phrases": ["DISCHARGE-Z"]},
        }
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "custom.xml"
            write_parameter_map(config, mapping)
            mapping = load_parameter_map(config)
            source = Path(temporary) / "synthetic.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "unixtime",
                        "custom.phase",
                        "epic.LAB.Kreatinine (umol/L)",
                        "q_umcg_aki_marco.adm_deceased",
                        "operation result",
                    ]
                )
                writer.writerows(
                    [
                        [100, "", 40, "", ""],
                        [200, "begin-x", "", "", ""],
                        [250, "", 250, 1, "severe problems"],
                        [300, "finish-y", "", "", ""],
                        [350, "", 40, 0, "optimal condition"],
                        [400, "DISCHARGE-Z", "", "", ""],
                    ]
                )
            pre = PreoperativeProcessor(mapping).process(source)
            self.assertEqual((pre.cutoff, pre.end_cutoff), (200, 300))
            self.assertEqual(pre.features["baseline_creatinine"], "40")
            post = PostoperativeProcessor(mapping).process(source, pre)
            self.assertEqual(post.measurements["postop_creatinine_72h"], "40")
            self.assertEqual(post.measurements["mortality_30d"], "0")
            self.assertEqual((post.target, post.target_status), (1, "success"))

    def test_xml_phase_items_can_use_different_event_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tree = ET.parse(ROOT / "config/pipeline.template.xml")
            catalog = tree.getroot().find("preprocessing/parameters")
            items = [catalog.find(tag + "/endEvent") for tag in ("preOperative", "duringOperative")]
            for event, field, phrase in zip(
                items, ("start_column", "end_column"), ("BEGIN-X", "FINISH-Y")
            ):
                for key, item_tag, value in (
                    ("fields", "field", field),
                    ("phrases", "phrase", phrase),
                ):
                    container = event.find(key)
                    container.clear()
                    ET.SubElement(container, item_tag).text = value
            config = root / "config.xml"
            tree.write(config, encoding="utf-8", xml_declaration=True)
            mapping = load_parameter_map(config)
            source = root / "synthetic.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["unixtime", "start_column", "end_column", "epic.LAB.Kreatinine (umol/L)"]
                )
                writer.writerows(
                    [
                        [100, "BEGIN-X", "", 10],
                        [120, "", "", 20],
                        [200, "", "FINISH-Y", 30],
                        [210, "", "", 40],
                    ]
                )
            pre = PreoperativeProcessor(mapping).process(source)
            self.assertEqual((pre.cutoff, pre.end_cutoff), (100, 200))
            with redirect_stderr(io.StringIO()), Logger(log_root=root / "logs") as logger:
                analyze_inputs([source], root / "results", config, logger, generate_plots=False)
            with (root / "results/duringoperative_measurements.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(next(csv.DictReader(stream))["during_creatinine"], "20")
            with (root / "results/postoperative_measurements.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(next(csv.DictReader(stream))["postop_creatinine_72h"], "40")

    def test_during_values_are_exported_only_between_configured_markers(self):
        mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["unixtime", "epic.EVENTS.EVENT", "epic.LAB.Kreatinine (umol/L)"])
                writer.writerows(
                    [
                        [100, "", 10],
                        [200, "Start surgery", 20],
                        [250, "", 30],
                        [300, "Stop anesthesie", 40],
                        [350, "", 50],
                    ]
                )
            with redirect_stderr(io.StringIO()), Logger(log_root=root / "logs") as logger:
                analyze_inputs([source], root / "results", mapping, logger, generate_plots=False)
            with (root / "results/duringoperative_measurements.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["during_creatinine"], "30")
            with (root / "results/patient_vectors.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                vector = next(csv.DictReader(stream))
            self.assertEqual(vector["baseline_creatinine"], "10")
            self.assertNotIn("during_creatinine", vector)
            with (root / "results/postoperative_measurements.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                postoperative = next(csv.DictReader(stream))
            self.assertEqual(postoperative["postop_creatinine_72h"], "50")


if __name__ == "__main__":
    unittest.main()
