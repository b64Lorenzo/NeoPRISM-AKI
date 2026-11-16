"""Plot selectors are independent from extraction/model parameter selection."""

import csv
import json
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from src.logging.logger import Logger
from src.pipeline.config import PipelineConfig
from src.pipeline.runner import run_pipeline
from src.preprocessing.parameters.parameter_map import load_parameter_map
from src.preprocessing.patient_pipeline import analyze_inputs
from src.preprocessing.plotting.distributions.plots import generate_parameter_distributions
from src.preprocessing.plotting.selection import PlotSelection
from src.utils.utils import _section

ROOT = Path(__file__).resolve().parents[1]


class PlotSelectionTests(unittest.TestCase):
    def test_each_plot_owns_its_selection_in_both_xml_files(self):
        for name in ("pipeline.xml", "pipeline.template.xml"):
            with self.subTest(name=name):
                plots = PipelineConfig.load(ROOT / "config" / name).preprocessing.plotting.as_dict()
                self.assertIsInstance(plots["distributions"], dict)
                for child, phase in (
                    ("distributions", "all"),
                    ("missingness_by_patient", "preoperative"),
                ):
                    self.assertEqual(plots[child]["phase"], phase)
                    self.assertEqual(plots[child]["parameter_ids"], [])
                    self.assertEqual(plots[child]["class_names"], [])
                self.assertEqual(plots["distributions"]["bins"], 20)
                self.assertNotIn("phase", plots)
                self.assertNotIn("bins", plots)

    def test_distributions_intersect_phase_parameter_and_class_filters(self):

        mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "patient_vectors.csv").write_text("patient_id,age\np1,10\n", encoding="utf-8")
            (output / "postoperative_measurements.csv").write_text(
                "patient_id,postop_creatinine_72h\np1,90\n", encoding="utf-8"
            )
            count = generate_parameter_distributions(
                output,
                mapping,
                phase="postoperative",
                parameter_ids=["age", "postop_creatinine_72h"],
                class_names=["epic.LAB"],
            )
            with (output / "parameter_distribution_summary.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(count, 1)
            self.assertEqual([row["parameter_id"] for row in rows], ["postop_creatinine_72h"])
            self.assertEqual(rows[0]["observed_count"], "1")
            self.assertEqual(rows[0]["mean"], "90")
            self.assertTrue((output / rows[0]["plot"]).is_file())

    def test_missingness_selects_across_phases_without_changing_extracted_columns(self):

        mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            source = project / "synthetic.csv"
            source.write_text(
                "unixtime,epic.EVENTS.EVENT,epic.DEMOGRAPHICS.age,epic.LAB.Kreatinine (umol/L)\n"
                "100,,10,40\n200,Start surgery,,\n250,,,60\n300,Stop anesthesie,,\n350,,,90\n",
                encoding="utf-8",
            )
            plot = project / "plots/missingness.html"
            selected = ["age", "sex", "during_creatinine", "postop_creatinine_72h"]
            with Logger(log_root=project / "logs") as logger:
                analyze_inputs(
                    [source],
                    project / "results",
                    mapping,
                    logger,
                    generate_plots=False,
                    generate_patient_missingness=True,
                    patient_missingness_output=plot,
                    patient_missingness_selection={"phase": "all", "parameter_ids": selected},
                )
            html = plot.read_text(encoding="utf-8")
            match = re.search(r'Plotly.newPlot\(\s*"[^"]+",\s*', html)
            assert match is not None, "No Plotly data in generated heatmap"
            traces = json.JSONDecoder().raw_decode(html[match.end() :])[0]
            self.assertEqual(traces[0]["x"], selected)
            self.assertEqual(traces[0]["z"], [[0.0, 1.0, 0.0, 0.0]])
            with (project / "results/patient_vectors.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                headers = next(csv.reader(stream))
            self.assertEqual(len(headers), 10)
            self.assertNotIn("during_creatinine", headers)
            self.assertNotIn("postop_creatinine_72h", headers)

    def test_phase_only_and_class_only_selection_preserve_catalog_order(self):

        mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
        for phase, expected in (
            ("preoperative", 9),
            ("duringoperative", 1),
            ("postoperative", 10),
            ("all", 20),
        ):
            with self.subTest(phase=phase):
                self.assertEqual(len(PlotSelection(phase, [], []).select(mapping)), expected)
        self.assertEqual(
            [
                item["id"]
                for item in PlotSelection("duringoperative", [], ["epic.LAB"]).select(mapping)
            ],
            ["during_creatinine"],
        )
        selected = PlotSelection("preoperative", ["sex", "age"], []).select(mapping)
        self.assertEqual([item["id"] for item in selected], ["age", "sex"])

    def test_invalid_selections_are_rejected_before_creating_plot_outputs(self):

        mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
        for phase, ids, classes in (
            ("all", ["unknown"], []),
            ("all", [], ["epic.UNKNOWN"]),
            ("postoperative", ["age"], []),
            ("all", ["age"], ["epic.LAB"]),
            ("unknown", [], []),
        ):
            with (
                self.subTest(phase=phase, ids=ids, classes=classes),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = Path(temporary) / "not-created"
                with self.assertRaises(ValueError):
                    generate_parameter_distributions(
                        output, mapping, phase=phase, parameter_ids=ids, class_names=classes
                    )
                self.assertFalse(output.exists())

    def test_malformed_selector_items_are_rejected_by_both_plot_parsers(self):

        for child in ("distributions", "missingnessByPatient"):
            for tag, value in (("name", "age"), ("id", " ")):
                with (
                    self.subTest(child=child, tag=tag, value=value),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
                    ET.SubElement(
                        _section(root, f"preprocessing/plotting/{child}/parameterIds"), tag
                    ).text = value
                    path = Path(temporary) / "pipeline.xml"
                    ET.ElementTree(root).write(path, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "parameterIds"):
                        PipelineConfig.load(path)

    def test_xml_runner_applies_independent_plot_filters_and_keeps_zero_present(self):

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            _section(root, "preprocessing/run").text = "true"
            _section(root, "preprocessing/inputDirectory").text = "data"
            _section(root, "preprocessing/plotting/run").text = "true"
            distributions = _section(root, "preprocessing/plotting/distributions")
            _section(distributions, "run").text = "true"
            _section(distributions, "phase").text = "postoperative"
            ET.SubElement(
                _section(distributions, "parameterIds"), "id"
            ).text = "postop_creatinine_72h"
            ET.SubElement(_section(distributions, "classNames"), "name").text = "epic.LAB"
            missingness = _section(root, "preprocessing/plotting/missingnessByPatient")
            _section(missingness, "run").text = "true"
            _section(missingness, "phase").text = "duringoperative"
            ET.SubElement(_section(missingness, "classNames"), "name").text = "epic.LAB"
            _section(missingness, "output").text = "results/custom_missingness.html"
            path = project / "config/pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8")
            source = project / "data/synthetic.csv"
            source.parent.mkdir()
            source.write_text(
                "unixtime,epic.EVENTS.EVENT,epic.DEMOGRAPHICS.age,epic.LAB.Kreatinine (umol/L)\n"
                "100,,10,40\n200,Start surgery,,\n250,,,0\n300,Stop anesthesie,,\n350,,,90\n",
                encoding="utf-8",
            )
            self.assertIn("preprocessing plots", run_pipeline(path))
            with (project / "results/parameter_distribution_summary.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(
                    [row["parameter_id"] for row in csv.DictReader(stream)],
                    ["postop_creatinine_72h"],
                )
            html = (project / "results/custom_missingness.html").read_text(encoding="utf-8")
            match = re.search(r'Plotly.newPlot\(\s*"[^"]+",\s*', html)
            assert match is not None
            traces = json.JSONDecoder().raw_decode(html[match.end() :])[0]
            self.assertEqual(traces[0]["x"], ["during_creatinine"])
            self.assertEqual(traces[0]["z"], [[0.0]])
