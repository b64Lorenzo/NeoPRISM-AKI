"""Contracts for component-owned XML parsing and the unified parameters node."""

import copy
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from src.utils.utils import _section

ROOT = Path(__file__).resolve().parents[1]


class ComponentConfigurationTests(unittest.TestCase):
    def test_both_xml_files_have_one_parameters_node_owning_selection(self):
        for name in ("pipeline.xml", "pipeline.template.xml"):
            with self.subTest(name=name):
                preprocessing = _section(
                    ET.parse(ROOT / "config" / name).getroot(), "preprocessing"
                )
                parameters = preprocessing.findall("parameters")
                self.assertEqual(len(parameters), 1)
                self.assertEqual(
                    [child.tag for child in parameters[0]],
                    ["settings", "preOperative", "duringOperative", "postOperative"],
                )
                self.assertIsNone(preprocessing.find("relevantParameters"))
                self.assertIsNone(preprocessing.find("parameterSelection"))

    def test_component_configs_are_typed_and_keep_the_dictionary_projection(self):
        from src.logging.config import LoggingConfig
        from src.pipeline.config import PipelineConfig, load_pipeline_config
        from src.preprocessing.config import PreprocessingConfig
        from src.preprocessing.parameters.config import ParametersConfig
        from src.preprocessing.parameters.settings import ParameterSettingsConfig
        from src.preprocessing.plotting.config import PreprocessingPlotConfig
        from src.preprocessing.plotting.missingness_by_patient.config import (
            MissingnessByPatientConfig,
        )
        from src.preprocessing.plotting.patient_timeline.config import PatientTimelineConfig
        from src.preprocessing.plotting.time_series.config import TimeSeriesPlotConfig

        path = ROOT / "config" / "pipeline.template.xml"
        config = PipelineConfig.load(path)
        self.assertIsInstance(config.logging, LoggingConfig)
        self.assertIsInstance(config.preprocessing, PreprocessingConfig)
        self.assertIsInstance(config.preprocessing.parameters, ParametersConfig)
        self.assertIsInstance(config.preprocessing.parameters.settings, ParameterSettingsConfig)
        plotting = config.preprocessing.plotting
        self.assertIsInstance(plotting, PreprocessingPlotConfig)
        self.assertIsInstance(plotting.time_series, TimeSeriesPlotConfig)
        self.assertIsInstance(plotting.patient_timeline, PatientTimelineConfig)
        self.assertIsInstance(plotting.missingness_by_patient, MissingnessByPatientConfig)
        self.assertEqual(config.as_dict(), load_pipeline_config(path))

    def test_catalog_entrypoints_return_the_same_unfiltered_definitions(self):
        from src.preprocessing.parameters.parameter_map import (
            load_parameter_map,
            load_parameter_map_element,
        )

        path = ROOT / "config" / "pipeline.template.xml"
        root = ET.parse(path).getroot()
        parameters = _section(root, "preprocessing/parameters")
        _section(parameters, "settings/selectionEnabled").text = "true"
        ET.SubElement(
            ET.SubElement(_section(parameters, "preOperative"), "includeIds"), "id"
        ).text = "age"
        expected = load_parameter_map(path)
        self.assertEqual(load_parameter_map_element(root), expected)
        self.assertEqual(load_parameter_map_element(parameters), expected)
        self.assertGreater(len(expected["parameters"]), 1)

    def test_runtime_rejects_ambiguous_or_obsolete_parameter_layouts(self):
        from src.pipeline.config import PipelineConfig

        original = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
        for case in (
            "duplicate",
            "missing",
            "mixed",
            "legacy",
            "missing-settings",
            "reordered",
            "old-selection",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = copy.deepcopy(original)
                preprocessing = _section(root, "preprocessing")
                parameters = _section(preprocessing, "parameters")
                if case == "duplicate":
                    preprocessing.append(copy.deepcopy(parameters))
                elif case == "missing":
                    preprocessing.remove(parameters)
                elif case == "mixed":
                    ET.SubElement(preprocessing, "parameterSelection")
                elif case == "legacy":
                    parameters.tag = "relevantParameters"
                elif case == "missing-settings":
                    parameters.remove(_section(parameters, "settings"))
                elif case == "reordered":
                    settings = _section(parameters, "settings")
                    parameters.remove(settings)
                    parameters.append(settings)
                else:
                    parameters.insert(0, ET.Element("selection"))
                path = Path(temporary) / "config" / "pipeline.xml"
                path.parent.mkdir()
                ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
                with self.assertRaisesRegex(ValueError, "parameters|settings"):
                    PipelineConfig.load(path)

    def test_catalog_loader_rejects_ambiguous_pipeline_parameter_sections(self):
        from src.preprocessing.parameters.parameter_map import load_parameter_map_element

        for case in ("duplicate-preprocessing", "duplicate-parameters", "legacy-sibling"):
            with self.subTest(case=case):
                root = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
                preprocessing = _section(root, "preprocessing")
                if case == "duplicate-preprocessing":
                    root.append(copy.deepcopy(preprocessing))
                elif case == "duplicate-parameters":
                    preprocessing.append(copy.deepcopy(_section(preprocessing, "parameters")))
                else:
                    ET.SubElement(preprocessing, "parameterSelection")
                with self.assertRaisesRegex(ValueError, "parameters|preprocessing"):
                    load_parameter_map_element(root)

    def test_selection_applies_only_when_enabled_and_does_not_mutate_the_catalog(self):
        from dataclasses import replace

        from src.pipeline.config import PipelineConfig

        parameters = PipelineConfig.load(
            ROOT / "config" / "pipeline.template.xml"
        ).preprocessing.parameters
        expected = copy.deepcopy(parameters.parameter_map)
        self.assertEqual(parameters.selected_map(), expected)
        phases = tuple(
            replace(
                phase,
                selected=phase.name == "Pre-operative",
                include_ids=["age"] if phase.name == "Pre-operative" else [],
            )
            for phase in parameters.phases
        )
        filtered = replace(
            parameters, settings=replace(parameters.settings, selection_enabled=True), phases=phases
        )
        self.assertEqual([item["id"] for item in filtered.selected_map()["parameters"]], ["age"])
        self.assertEqual(parameters.parameter_map, expected)
        self.assertEqual(
            replace(
                filtered, settings=replace(filtered.settings, selection_enabled=False)
            ).selected_map(),
            expected,
        )
