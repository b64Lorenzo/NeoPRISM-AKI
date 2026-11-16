"""Synthetic checks for phase-relative time-series extraction."""
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

ROOT = Path(__file__).resolve().parents[1]


class PhaseTimeSeriesTests(unittest.TestCase):
    def test_active_and_template_xml_configure_time_series_switch(self):
        expected = {"pipeline.xml": True, "pipeline.template.xml": False}
        for name, enabled in expected.items():
            with self.subTest(name=name):
                self.assertIs(load_parameter_map(ROOT / "config" / name)["export_time_series"], enabled)

    def test_non_boolean_time_series_switch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
            mapping["export_time_series"] = "true"
            config = Path(temporary) / "invalid.xml"
            write_parameter_map(config, mapping)
            with self.assertRaisesRegex(ValueError, "export_time_series.*boolean"):
                load_parameter_map(config)

    def test_numbered_parts_cross_midnight_without_dates_or_window_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
            mapping["export_time_series"] = True
            inputs = root / "inputs"
            inputs.mkdir()
            with (inputs / "ref_0002.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["datetime", "epic.EVENTS.EVENT", "epic.LAB.Kreatinine (umol/L)"])
                writer.writerows([
                    ["2026-03-02T23:59:00Z", "Start surgery", 2],
                    ["2026-03-02T23:59:30Z", "", 3],
                ])
            with (inputs / "ref_0002_1.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["epic.LAB.Kreatinine (umol/L)", "epic.EVENTS.EVENT", "datetime"])
                writer.writerows([
                    [4, "Stop anesthesie", "2026-03-03T00:00:00Z"],
                    [5, "", "2026-03-03T00:01:00Z"],
                    [6, "", "2026-03-06T01:00:00Z"],
                ])
            with redirect_stderr(io.StringIO()), Logger(log_root=root / "logs") as logger:
                analyze_inputs([inputs], root / "results", mapping, logger, generate_plots=False)
            exported = (root / "results/time_series.csv").read_text(encoding="utf-8")
            rows = list(csv.DictReader(io.StringIO(exported)))
            self.assertEqual([(r["phase"], r["t_seconds"], r["value"]) for r in rows], [
                ("During-operative", "0", "2"), ("During-operative", "30", "3"),
                ("Post-operative", "0", "4"), ("Post-operative", "60", "5"),
            ])
            self.assertEqual({r["patient_id"] for r in rows}, {"patient_001"})
            self.assertNotIn("2026", exported)

    def test_missing_end_does_not_reset_clock_at_first_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
            mapping["export_time_series"] = True
            source = root / "synthetic.csv"
            source.write_text(
                "unixtime,epic.EVENTS.EVENT,epic.LAB.Kreatinine (umol/L)\n"
                "100,Start surgery,20\n200,,30\n", encoding="utf-8",
            )
            with redirect_stderr(io.StringIO()), Logger(log_root=root / "logs") as logger:
                analyze_inputs([source], root / "results", mapping, logger, generate_plots=False)
            with (root / "results/time_series.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])

    def test_first_observation_does_not_become_time_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
            mapping["export_time_series"] = True
            source = root / "synthetic.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["unixtime", "epic.EVENTS.EVENT", "epic.LAB.Kreatinine (umol/L)"])
                writer.writerows([
                    [100, "Start surgery", ""], [115, "", "0"], [120, "", "NA"],
                    [200, "Stop anesthesie", ""], [220, "", "7 | note"],
                    ["", "", "9"], [240, "", "not-a-number"],
                ])
            with redirect_stderr(io.StringIO()), Logger(log_root=root / "logs") as logger:
                analyze_inputs([source], root / "results", mapping, logger, generate_plots=False)
            with (root / "results/time_series.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([(r["phase"], r["t_seconds"], r["value"]) for r in rows], [
                ("During-operative", "15", "0"), ("Post-operative", "20", "7"),
            ])

    def test_enabled_time_series_has_independent_during_and_postoperative_clocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            xml = ET.parse(ROOT / "config/pipeline.template.xml")
            catalog = xml.getroot().find("preprocessing/parameters")
            settings = catalog.find("settings")
            settings.find("exportTimeSeries").text = "true"
            config = root / "config.xml"
            xml.write(config, encoding="utf-8", xml_declaration=True)
            source = root / "synthetic.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["unixtime", "epic.EVENTS.EVENT", "epic.LAB.Kreatinine (umol/L)"])
                writer.writerows([
                    [100, "", 10], [200, "Start surgery", 20], [260, "", 30],
                    [300, "Stop anesthesie", 40], [360, "", 50],
                ])
            with redirect_stderr(io.StringIO()), Logger(log_root=root / "logs") as logger:
                analyze_inputs([source], root / "results", config, logger, generate_plots=False)
            with (root / "results/time_series.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["patient_id", "phase", "parameter_id", "t_seconds", "value"])
                rows = list(reader)
            self.assertEqual(
                [(row["phase"], row["parameter_id"], row["t_seconds"], row["value"]) for row in rows],
                [("During-operative", "during_creatinine", "0", "20"),
                 ("During-operative", "during_creatinine", "60", "30"),
                 ("Post-operative", "postop_creatinine_72h", "0", "40"),
                 ("Post-operative", "postop_creatinine_72h", "60", "50")],
            )
            self.assertEqual({row["patient_id"] for row in rows}, {"patient_001"})


if __name__ == "__main__":
    unittest.main()
