"""Regression checks for the reorganized package and shared XML helpers."""

import importlib
import tempfile
import typing
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from src.utils.utils import _section


class XmlUtilityTests(unittest.TestCase):
    def test_shared_helpers_import_with_resolvable_type_hints(self):
        module = importlib.import_module("src.utils.utils")
        for name in ("_required", "_boolean", "_path", "_ids"):
            self.assertTrue(typing.get_type_hints(getattr(module, name)))

    def test_required_value_is_stripped_and_empty_values_are_rejected(self):
        from src.utils.utils import _required

        root = ET.fromstring("<config><value> text </value><empty> </empty></config>")
        self.assertEqual(_required(root, "value"), "text")
        for tag in ("empty", "missing"):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, tag):
                _required(root, tag)

    def test_boolean_values_are_validated(self):
        from src.utils.utils import _boolean

        root = ET.Element("config")
        flag = ET.SubElement(root, "run")
        for value, expected in (
            (" true ", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("0", False),
        ):
            with self.subTest(value=value):
                flag.text = value
                self.assertIs(_boolean(root, "run"), expected)
        flag.text = "yes"
        with self.assertRaisesRegex(ValueError, "run must be true or false"):
            _boolean(root, "run")

    def test_paths_resolve_relative_to_the_project(self):
        from src.utils.utils import _path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(_path(root, "config/../results"), root / "results")
            self.assertEqual(_path(root, str(root / "logs")), root / "logs")

    def test_id_lists_allow_empty_containers_and_reject_duplicates(self):
        from src.utils.utils import _ids

        root = ET.fromstring("<config><empty/><ids><id> age </id><id>sex</id></ids></config>")
        self.assertEqual(_ids(root, "empty"), [])
        self.assertEqual(_ids(root, "ids"), ["age", "sex"])
        ET.SubElement(_section(root, "ids"), "id").text = "age"
        with self.assertRaisesRegex(ValueError, "Duplicate value in ids"):
            _ids(root, "ids")
        with self.assertRaisesRegex(ValueError, "Missing required XML section: absent"):
            _ids(root, "absent")
