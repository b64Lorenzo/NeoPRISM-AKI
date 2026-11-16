"""Contracts for the compact input directory and phase-owned settings."""

import copy
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from src.pipeline.config import PipelineConfig
from src.utils.utils import _section

ROOT = Path(__file__).resolve().parents[1]
PHASE_TAGS = ("preOperative", "duringOperative", "postOperative")


class CompactConfigurationTests(unittest.TestCase):
    def test_xml_uses_one_input_directory_and_named_phases(self):
        for name in ("pipeline.xml", "pipeline.template.xml"):
            with self.subTest(name=name):
                preprocessing = _section(
                    ET.parse(ROOT / "config" / name).getroot(), "preprocessing"
                )
                self.assertIsNotNone(preprocessing.find("inputDirectory"))
                self.assertIsNone(preprocessing.find("inputs"))
                parameters = _section(preprocessing, "parameters")
                self.assertEqual([node.tag for node in parameters], ["settings", *PHASE_TAGS])
                self.assertIsNone(parameters.find("selection"))
                settings = _section(parameters, "settings")
                self.assertIsNotNone(settings.find("selectionEnabled"))
                self.assertIsNotNone(settings.find("exportTimeSeries"))
                self.assertEqual(settings.findall("entry"), [])
                for tag in PHASE_TAGS:
                    self.assertIsNotNone(_section(parameters, tag).find("endEvent"))

    def test_workbook_provenance_is_not_required_or_loaded(self):
        for name in ("pipeline.xml", "pipeline.template.xml"):
            config = PipelineConfig.load(ROOT / "config" / name)
            mapping = config.preprocessing.parameters.parameter_map
            self.assertNotIn("source_workbook", mapping)
            self.assertNotIn("source_sheet", mapping)
            for parameter in mapping["parameters"]:
                self.assertFalse(
                    any("workbook" in key or key == "source_sheet" for key in parameter)
                )

    def test_phase_filters_are_owned_by_their_phase_and_do_not_mutate_the_catalog(self):
        from dataclasses import replace

        config = PipelineConfig.load(ROOT / "config" / "pipeline.template.xml")
        parameters = config.preprocessing.parameters
        original = copy.deepcopy(parameters.parameter_map)
        phases = tuple(
            replace(phase, include_ids=["age"]) if phase.name == "Pre-operative" else phase
            for phase in parameters.phases
        )
        filtered = replace(
            parameters, settings=replace(parameters.settings, selection_enabled=True), phases=phases
        )
        selected = filtered.selected_map()
        self.assertEqual(
            [item["id"] for item in selected["parameters"] if item["phase"] == "Pre-operative"],
            ["age"],
        )
        self.assertEqual(
            sum(item["phase"] == "Post-operative" for item in selected["parameters"]), 10
        )
        self.assertEqual(parameters.parameter_map, original)
        self.assertEqual(
            replace(
                filtered, settings=replace(filtered.settings, selection_enabled=False)
            ).selected_map(),
            original,
        )

    def test_xml_phase_filters_execute_through_the_pipeline(self):
        from src.pipeline.runner import run_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
            parameters = _section(root, "preprocessing/parameters")
            _section(parameters, "settings/selectionEnabled").text = "true"
            ET.SubElement(
                ET.SubElement(_section(parameters, "preOperative"), "includeIds"), "id"
            ).text = "age"
            for tag in ("duringOperative", "postOperative"):
                _section(parameters, tag).set("selected", "false")
            _section(root, "preprocessing/parameterDescriptions/run").text = "true"
            _section(
                root, "preprocessing/parameterDescriptions/output"
            ).text = "results/reference.md"
            path = project / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            self.assertEqual(run_pipeline(path), "parameter reference")
            reference = (project / "results/reference.md").read_text(encoding="utf-8")
            self.assertIn("| age |", reference)
            self.assertNotIn("| sex |", reference)
            self.assertFalse((project / "results/patient_vectors.csv").exists())

    def test_phase_filters_reject_cross_phase_ids_overlap_and_no_selected_phase(self):
        from dataclasses import replace

        parameters = PipelineConfig.load(
            ROOT / "config" / "pipeline.template.xml"
        ).preprocessing.parameters
        for case in ("cross-phase", "overlap", "no-phase"):
            with self.subTest(case=case):
                phases = list(parameters.phases)
                if case == "cross-phase":
                    phases[0] = replace(phases[0], include_ids=["recorded_death_days"])
                elif case == "overlap":
                    phases[0] = replace(phases[0], include_ids=["age"], exclude_ids=["age"])
                else:
                    phases = [replace(phase, selected=False) for phase in phases]
                configured = replace(
                    parameters,
                    settings=replace(parameters.settings, selection_enabled=True),
                    phases=tuple(phases),
                )
                with self.assertRaisesRegex(ValueError, "phase|included and excluded"):
                    configured.selected_map()

    def test_empty_and_duplicate_input_directories_are_rejected(self):
        for case in ("empty", "duplicate"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
                preprocessing = _section(root, "preprocessing")
                directory = _section(preprocessing, "inputDirectory")
                if case == "empty":
                    directory.text = " "
                else:
                    preprocessing.append(copy.deepcopy(directory))
                path = Path(temporary) / "config" / "pipeline.xml"
                path.parent.mkdir()
                ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
                with self.assertRaisesRegex(ValueError, "inputDirectory"):
                    PipelineConfig.load(path)
