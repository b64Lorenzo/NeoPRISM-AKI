"""Optional XSD checks complement runtime semantic validation."""

import copy
import importlib.util
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from src.preprocessing.parameters.parameter_map import load_parameter_map, write_parameter_map
from src.utils.utils import _section

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("lxml"), "optional lxml development dependency")
class XmlSchemaTests(unittest.TestCase):
    def setUp(self):
        from lxml import etree

        self.etree = etree
        self.schema = etree.XMLSchema(etree.parse(str(ROOT / "config" / "pipeline.xsd")))

    def test_checked_in_configs_validate_against_the_single_schema(self):
        for name in ("pipeline.xml", "pipeline.template.xml"):
            with self.subTest(name=name):
                self.schema.assertValid(self.etree.parse(str(ROOT / "config" / name)))

    def test_schema_rejects_malformed_combined_parameter_sections(self):
        original = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
        for case in (
            "missing-settings",
            "duplicate-settings",
            "reordered",
            "legacy-sibling",
            "duplicate-parameters",
            "format-version",
            "old-selection",
            "invalid-selected",
        ):
            with self.subTest(case=case):
                root = copy.deepcopy(original)
                preprocessing = _section(root, "preprocessing")
                parameters = _section(preprocessing, "parameters")
                settings = _section(parameters, "settings")
                if case == "missing-settings":
                    parameters.remove(settings)
                elif case == "duplicate-settings":
                    parameters.insert(1, copy.deepcopy(settings))
                elif case == "reordered":
                    parameters.remove(settings)
                    parameters.append(settings)
                elif case == "legacy-sibling":
                    ET.SubElement(preprocessing, "parameterSelection")
                elif case == "duplicate-parameters":
                    preprocessing.append(copy.deepcopy(parameters))
                elif case == "format-version":
                    parameters.set("formatVersion", "2")
                elif case == "old-selection":
                    parameters.insert(0, ET.Element("selection"))
                else:
                    _section(parameters, "preOperative").set("selected", "yes")
                self.assertFalse(self.schema.validate(self.etree.fromstring(ET.tostring(root))))

    def test_standalone_catalog_writer_remains_schema_valid(self):
        expected = load_parameter_map(ROOT / "config" / "pipeline.template.xml")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "map.xml"
            write_parameter_map(path, expected)
            self.schema.assertValid(self.etree.parse(str(path)))
            self.assertEqual(load_parameter_map(path), expected)
