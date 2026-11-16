"""Import and real CLI smoke checks without reading cohort inputs."""

import importlib
import io
import pkgutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path

import src
from src.utils.utils import _section

ROOT = Path(__file__).resolve().parents[1]


class PackageImportTests(unittest.TestCase):
    def test_every_source_module_imports(self):
        for module in pkgutil.walk_packages(src.__path__, "src."):
            with self.subTest(module=module.name):
                importlib.import_module(module.name)

    def test_cli_run_reports_a_disabled_pipeline_without_an_unbound_logger(self):
        from src.cli import main

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            path = project / "config" / "pipeline.xml"
            path.parent.mkdir()
            root = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
            _section(root, "logging/directory").text = str(project / "logs")
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["run", "--config", str(path)])
            self.assertEqual(status, 0)
            self.assertIn("XML pipeline complete: no stages enabled", output.getvalue())
            self.assertEqual(len(list((project / "logs").glob("*/run.log"))), 1)
