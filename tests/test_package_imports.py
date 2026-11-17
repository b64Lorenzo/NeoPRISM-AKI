"""Import and real CLI smoke checks without reading cohort inputs."""

import importlib
import pkgutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import src
from src.utils.utils import _section

ROOT = Path(__file__).resolve().parents[1]


class PackageImportTests(unittest.TestCase):
    def test_every_source_module_imports(self):
        for module in pkgutil.walk_packages(src.__path__, "src."):
            with self.subTest(module=module.name):
                importlib.import_module(module.name)

    def test_cli_help_lists_only_run_subcommand(self):
        from src.cli import build_parser

        help_text = build_parser().format_help()
        self.assertIn("usage: pattern-recognition [-h] {run} ...", help_text)

    def test_cli_run_logs_disabled_pipeline_completion(self):
        from src.cli import main

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            path = project / "config" / "pipeline.xml"
            path.parent.mkdir()
            root = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
            _section(root, "logging/directory").text = str(project / "logs")
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            status = main(["run", "--config", str(path)])
            self.assertEqual(status, 0)
            logs = list((project / "logs").glob("*/run.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn(
                "XML pipeline complete: no stages enabled",
                logs[0].read_text(encoding="utf-8"),
            )
