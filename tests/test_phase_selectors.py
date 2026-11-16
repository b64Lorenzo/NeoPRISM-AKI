"""Synthetic configuration tests for phase-owned parameter definitions."""

import csv
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

from src.pipeline.config import load_pipeline_config
from src.pipeline.runner import run_pipeline
from src.preprocessing.patient_pipeline import analyze_inputs
from src.preprocessing.postoperative.outcome_rules import OutcomeClassifier
from src.preprocessing.postoperative.processor import PostoperativeProcessor
from src.preprocessing.preoperative.processor import PreoperativeProcessor

ROOT = Path(__file__).resolve().parents[1]


class PhaseSelectorTests(unittest.TestCase):
    @staticmethod
    def empty_phases(catalog):
        pre, during, post = (catalog.find(tag) for tag in ("preOperative", "duringOperative", "postOperative"))
        for section in (pre, during, post):
            for child in list(section):
                if child.tag not in {"endEvent", "includeIds", "excludeIds"}:
                    section.remove(child)
        return pre, during, post

    @staticmethod
    def parameter(catalog, phase, parameter_id):
        return next(p for p in catalog.find(phase).iter("parameter") if p.get("id") == parameter_id)

    @staticmethod
    def keep_parameters(catalog, ids):
        for section in (catalog.find(tag) for tag in ("preOperative", "duringOperative", "postOperative")):
            for group in list(section):
                if group.tag == "class":
                    for parameter in list(group):
                        if parameter.tag == "parameter" and parameter.get("id") not in ids:
                            group.remove(parameter)
                    if not group.findall("parameter"):
                        section.remove(group)
                elif group.tag not in {"endEvent", "includeIds", "excludeIds"} and group.get("id") not in ids:
                    section.remove(group)

    def test_class_expands_only_catalogued_parameters_of_its_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            catalog = root.find("preprocessing/parameters")
            ids = {"age", "sex", "weight", "height", "bmi", "asa", "recorded_death_days"}
            self.keep_parameters(catalog, ids)
            path = Path(temporary) / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            mapping = load_pipeline_config(path)["preprocessing"]["parameter_map"]
            self.assertEqual([item["id"] for item in mapping["parameters"]],
                             ["age", "sex", "weight", "height", "bmi", "asa", "recorded_death_days"])
            self.assertEqual([item["id"] for item in mapping["postoperative_measurements"]],
                             ["recorded_death_days"])

    def test_post_outcome_rule_is_read_from_its_definition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            catalog = root.find("preprocessing/parameters")
            death = self.parameter(catalog, "postOperative", "recorded_death_days")
            death.find("entry[@name='outcome_criteria']/object/entry[@name='failure']/object/entry[@name='value']/number").text = "2"
            self.keep_parameters(catalog, {"recorded_death_days"})
            path = Path(temporary) / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            classifier = OutcomeClassifier(load_pipeline_config(path)["preprocessing"]["parameter_map"])
            self.assertEqual(classifier.classify({"recorded_death_days": 2}).target, -1)
            self.assertIsNone(classifier.classify({"recorded_death_days": 1}).target)

    def test_post_class_nonempty_failure_only_uses_selected_fields_after_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            catalog = root.find("preprocessing/parameters")
            self.keep_parameters(catalog, {"postop_urine_72h"})
            catalog.find("postOperative/class[@name='epic.FLUIDS_OUT']").set("failure", "nonempty")
            config = Path(temporary) / "config" / "pipeline.xml"
            config.parent.mkdir()
            ET.ElementTree(root).write(config, encoding="utf-8", xml_declaration=True)
            mapping = load_pipeline_config(config)["preprocessing"]["parameter_map"]
            mapping["missing_value_tokens"] = ["NA"]
            self.assertEqual(mapping["post_class_failures"], ["epic.FLUIDS_OUT"])
            self.assertEqual([p["id"] for p in mapping["parameters"]], ["postop_urine_72h"])
            mapping["parameters"][0]["outcome_criteria"]["success"] = {"operator": "equals", "value": 0}
            source = Path(temporary) / "synthetic.csv"
            cases = (("5", "0", "300", -1), ("5", "NA", "300", None),
                     ("", "", "300", None), ("", "0", "300000", None))
            for before, after, after_time, expected in cases:
                with self.subTest(before=before, after=after, after_time=after_time):
                    with source.open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(["unixtime", "event", "epic.FLUIDS_OUT.Urine (ml)", "epic.FLUIDS_OUT.unmapped"])
                        writer.writerow(["100", "", before, ""])
                        writer.writerow(["200", "start surgery", "", ""])
                        writer.writerow(["250", "Stop anesthesie", "", ""])
                        writer.writerow([after_time, "", after, "unexpected value"])
                    pre_result = PreoperativeProcessor(mapping).process(source)
                    result = PostoperativeProcessor(mapping).process(source, pre_result)
                    self.assertEqual(result.target, expected)
                    self.assertEqual(result.target_status, "failure" if expected == -1 else "unclassified")
    def test_missing_end_event_leaves_during_and_postoperative_fields_blank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_pipeline_config(ROOT / "config/pipeline.template.xml")
            mapping = config["preprocessing"]["parameter_map"]
            mapping["parameters"].append({
                "id": "during_oxygen", "phase": "During-operative", "parameter_type": "numeric",
                "source_fields": ["synthetic.oxygen"],
            })
            source = root / "synthetic.csv"
            source.write_text("unixtime,event,synthetic.oxygen,operation result\n100,,5,\n200,start surgery,10,\n300,,20,severe problems\n", encoding="utf-8")
            from src.logging.logger import Logger
            with Logger(log_root=root / "logs") as logger:
                analyze_inputs([source], root / "results", mapping, logger, generate_plots=False)
            with (root / "results/duringoperative_measurements.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.DictReader(stream))["during_oxygen"], "")
            with (root / "results/patient_outcomes.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.DictReader(stream))["target_status"], "unclassified")
            self.assertIn("Patients without a matched operation end after start: 1",
                          (root / "results/extraction_report.md").read_text(encoding="utf-8"))

    def test_during_phase_definition_loads_for_header_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            catalog = root.find("preprocessing/parameters")
            during_parameter = deepcopy(self.parameter(catalog, "preOperative", "age"))
            during_parameter.set("id", "during_age")
            catalog.find("duringOperative").append(during_parameter)
            self.keep_parameters(catalog, {"during_age"})
            path = Path(temporary) / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            mapping = load_pipeline_config(path)["preprocessing"]["parameter_map"]
            self.assertEqual([p["id"] for p in mapping["parameters"]], ["during_age"])
    def test_xml_runner_extracts_during_parameters_and_reports_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            root.find("preprocessing/run").text = "true"
            root.find("preprocessing/csvHeaderDescriptions/run").text = "true"
            root.find("preprocessing/inputDirectory").text = "data/patients"
            catalog = root.find("preprocessing/parameters")
            during_parameter = deepcopy(self.parameter(catalog, "preOperative", "age"))
            during_parameter.set("id", "during_age")
            catalog.find("duringOperative").append(during_parameter)
            path = project / "config/pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            source = project / "data/patients/case.csv"
            source.parent.mkdir(parents=True)
            source.write_text(
                "unixtime,epic.EVENTS.EVENT,epic.DEMOGRAPHICS.age\n"
                "100,,5\n200,Start surgery,\n250,,10\n300,Stop anesthesie,\n350,,20\n",
                encoding="utf-8",
            )
            self.assertIn("preprocessing", run_pipeline(path))
            with (project / "results/duringoperative_measurements.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.DictReader(stream))["during_age"], "10")
            summary = (project / "results/csv_header_summary.md").read_text(encoding="utf-8")
            during = summary.split("## During-operative headers", 1)[1].split("## Post-operative headers", 1)[0]
            self.assertIn("| epic.DEMOGRAPHICS.age | epic.DEMOGRAPHICS | 1 |", during)
            self.assertIn("| epic.DEMOGRAPHICS | 1 |", during)
            self.assertTrue((project / "results/patient_vectors.csv").exists())
            root.find("preprocessing/plotting/run").text = "true"
            root.find("preprocessing/plotting/distributions").text = "true"
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            self.assertIn("preprocessing plots", run_pipeline(path))
            self.assertTrue((project / "results/parameter_distributions/duringoperative_during_age.html").is_file())

    def test_xml_runner_uses_patient_missingness_plot_setting(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            preprocessing = root.find("preprocessing")
            preprocessing.find("run").text = "true"
            preprocessing.find("inputDirectory").text = "data/patients"
            preprocessing.find("outputDirectory").text = "results"
            plotting = preprocessing.find("plotting")
            plotting.find("run").text = "true"
            missingness = plotting.find("missingnessByPatient")
            missingness.find("run").text = "true"
            missingness.find("output").text = "results/custom_missingness.html"

            config_path = project / "config" / "pipeline.xml"
            config_path.parent.mkdir()
            ET.ElementTree(root).write(config_path, encoding="utf-8", xml_declaration=True)
            source = project / "data" / "patients" / "synthetic.csv"
            source.parent.mkdir(parents=True)
            source.write_text(
                "unixtime,epic.EVENTS.EVENT,epic.DEMOGRAPHICS.age\n"
                "100,,5\n200,Start surgery,\n250,,10\n300,Stop anesthesie,\n",
                encoding="utf-8",
            )

            run_pipeline(config_path)

            custom_plot = project / "results" / "custom_missingness.html"
            self.assertTrue(custom_plot.is_file())
            html = custom_plot.read_text(encoding="utf-8")
            self.assertIn("Plotly.newPlot", html)
            self.assertIn("patient_001", html)
            self.assertFalse((project / "results/preoperative_missingness_heatmap.html").exists())

            custom_plot.unlink()
            missingness.find("run").text = "false"
            ET.ElementTree(root).write(config_path, encoding="utf-8", xml_declaration=True)
            run_pipeline(config_path)
            self.assertFalse(custom_plot.exists())
            self.assertFalse((project / "results/preoperative_missingness_heatmap.html").exists())

    def test_xml_runner_generates_configured_time_series_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            preprocessing = root.find("preprocessing")
            preprocessing.find("run").text = "true"
            preprocessing.find("inputDirectory").text = "data/patients"
            preprocessing.find("outputDirectory").text = "results"
            preprocessing.find("plotting/run").text = "false"
            preprocessing.find("plotting/distributions").text = "false"
            settings = preprocessing.find("parameters/settings")
            settings.find("exportTimeSeries").text = "true"
            time_series = preprocessing.find("plotting/timeSeries")
            if time_series is None:
                time_series = ET.SubElement(preprocessing.find("plotting"), "timeSeries")
            for child in list(time_series):
                time_series.remove(child)
            ET.SubElement(time_series, "run").text = "true"
            ET.SubElement(time_series, "groupBy").text = "class"
            parameter_ids = ET.SubElement(time_series, "parameterIds")
            ET.SubElement(parameter_ids, "id").text = "during_creatinine"
            class_names = ET.SubElement(time_series, "classNames")
            ET.SubElement(class_names, "name").text = "epic.LAB"
            ET.SubElement(time_series, "patientIds")
            ET.SubElement(time_series, "outputDirectory").text = "results/configured_time_series_plots"

            path = project / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            source = project / "data/patients/case.csv"
            source.parent.mkdir(parents=True)
            source.write_text(
                "unixtime,epic.EVENTS.EVENT,epic.LAB.Kreatinine (umol/L)\n"
                "100,,40\n200,Start surgery,\n250,,50\n300,Stop anesthesie,\n",
                encoding="utf-8",
            )

            completed = run_pipeline(path)

            self.assertIn("time-series plots", completed)
            self.assertTrue((project / "results/time_series.csv").is_file())
            self.assertTrue((project / "results/configured_time_series_plots/epic.LAB.html").is_file())

    def test_xml_runner_plots_existing_time_series_without_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            time_series = root.find("preprocessing/plotting/timeSeries")
            time_series.find("run").text = "true"
            time_series.find("groupBy").text = "parameter"
            ET.SubElement(time_series.find("parameterIds"), "id").text = "during_creatinine"
            time_series.find("outputDirectory").text = "results/time_series_plots"
            path = project / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

            results = project / "results"
            results.mkdir()
            with (results / "time_series.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["patient_id", "phase", "parameter_id", "t_seconds", "value"],
                )
                writer.writeheader()
                writer.writerow({
                    "patient_id": "patient_001", "phase": "During-operative",
                    "parameter_id": "during_creatinine", "t_seconds": "10", "value": "45",
                })

            completed = run_pipeline(path)

            self.assertEqual(completed, "time-series plots")
            self.assertTrue((results / "time_series_plots/during_creatinine.html").is_file())
            self.assertFalse((results / "patient_vectors.csv").exists())

    def test_empty_phase_selections_still_allow_header_only_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            catalog = root.find("preprocessing/parameters")
            self.empty_phases(catalog)
            path = Path(temporary) / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            mapping = load_pipeline_config(path)["preprocessing"]["parameter_map"]
            self.assertEqual(mapping["parameters"], [])

    def test_shared_settings_cannot_redefine_parameter_catalog(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            settings = root.find("preprocessing/parameters/settings")
            ET.SubElement(ET.SubElement(settings, "entry", {"name": "parameters"}), "array")
            path = Path(temporary) / "config/pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "Define parameters and measurements in their phase"):
                load_pipeline_config(path)

    def test_class_group_rejects_unrelated_parameter_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            catalog = root.find("preprocessing/parameters")
            age = self.parameter(catalog, "preOperative", "age")
            age.find("entry[@name='source_fields']/array/item/string").text = "epic.LAB.age"
            path = Path(temporary) / "config/pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "no source field in epic.DEMOGRAPHICS"):
                load_pipeline_config(path)

    def test_null_post_measurement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            catalog = root.find("preprocessing/parameters")
            parameter = self.parameter(catalog, "postOperative", "recorded_death_days")
            parameter.attrib.pop("measurementOrder")
            measurement = parameter.find("entry[@name='measurement']")
            measurement.remove(measurement[0])
            ET.SubElement(measurement, "null")
            path = Path(temporary) / "config/pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "Invalid post-operative measurement"):
                load_pipeline_config(path)

    def test_duplicate_parameter_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            catalog = root.find("preprocessing/parameters")
            self.parameter(catalog, "postOperative", "ventilation_duration").set("id", "age")
            path = Path(temporary) / "config/pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "Parameter IDs must be present and unique"):
                load_pipeline_config(path)

    def test_embedded_map_requires_all_three_phase_sections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            catalog = root.find("preprocessing/parameters")
            for tag in ("preOperative", "duringOperative", "postOperative"):
                catalog.remove(catalog.find(tag))
            path = Path(temporary) / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "preOperative, duringOperative, and postOperative"):
                load_pipeline_config(path)

    @unittest.skipUnless(shutil.which("xmllint"), "xmllint is not installed")
    def test_schema_accepts_phase_sections_and_rejects_nonpost_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            catalog = root.find("preprocessing/parameters")
            pre = catalog.find("preOperative")
            self.keep_parameters(catalog, {"age", "recorded_death_days"})
            path = Path(temporary) / "synthetic.xml"
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            command = ["xmllint", "--noout", "--schema", str(ROOT / "config/pipeline.xsd"), str(path)]
            valid = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            root.find("preprocessing/plotting/timeSeries/groupBy").text = "patient"
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            invalid_group = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(invalid_group.returncode, 0)
            root.find("preprocessing/plotting/timeSeries/groupBy").text = "parameter"
            pre.find("class").set("failure", "nonempty")
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            invalid = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == "__main__":
    unittest.main()
