"""Synthetic tests for time-series plot selection and comparisons."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from src.parameter_map import load_parameter_map
from src.plotly_output import write_plotly_html
from src.time_series_plots import TimeSeriesPlotter
from src.cli import main

ROOT = Path(__file__).resolve().parents[1]


class TimeSeriesPlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mapping = load_parameter_map(ROOT / "config" / "pipeline.template.xml")
        self.input = self.root / "time_series.csv"
        with self.input.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["patient_id", "phase", "parameter_id", "t_seconds", "value"],
            )
            writer.writeheader()
            writer.writerows([
                {"patient_id": "patient_001", "phase": "During-operative", "parameter_id": "during_creatinine", "t_seconds": "0", "value": "10"},
                {"patient_id": "patient_001", "phase": "During-operative", "parameter_id": "during_creatinine", "t_seconds": "10", "value": "12"},
                {"patient_id": "patient_002", "phase": "During-operative", "parameter_id": "during_creatinine", "t_seconds": "0", "value": "11"},
                {"patient_id": "patient_002", "phase": "During-operative", "parameter_id": "during_creatinine", "t_seconds": "10", "value": "13"},
                {"patient_id": "patient_001", "phase": "Post-operative", "parameter_id": "postop_creatinine_72h", "t_seconds": "0", "value": "14"},
                {"patient_id": "patient_002", "phase": "Post-operative", "parameter_id": "postop_creatinine_72h", "t_seconds": "5", "value": "15"},
                {"patient_id": "patient_001", "phase": "Post-operative", "parameter_id": "postop_urine_72h", "t_seconds": "0", "value": "2"},
            ])

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_time_series_parameters_plot_each_patient_for_comparison(self):
        output = self.root / "plots"
        result = TimeSeriesPlotter(self.mapping).generate(self.input, output)

        self.assertEqual(result["plot_count"], 3)
        self.assertEqual(result["parameter_count"], 3)
        self.assertEqual(result["patient_count"], 2)
        self.assertEqual(result["observation_count"], 7)
        self.assertEqual({path.name for path in result["plots"]}, {
            "during_creatinine.html", "postop_creatinine_72h.html", "postop_urine_72h.html",
        })
        self.assertTrue(all(path.is_file() and path.stat().st_size > 1000 for path in result["plots"]))

    def test_dark_theme_applies_to_time_series_charts(self):
        with patch(
            "src.time_series_plots.write_plotly_html",
            wraps=write_plotly_html,
        ) as write_plot:
            TimeSeriesPlotter(self.mapping).generate(
                self.input,
                self.root / "dark-plots",
                parameter_ids=["during_creatinine"],
                theme="dark",
            )

        self.assertEqual(
            write_plot.call_args.args[0].layout.template.layout.paper_bgcolor,
            "rgb(17,17,17)",
        )

    def test_long_postoperative_times_are_plotted_as_hours(self):
        source = self.root / "long-time-series.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["patient_id", "phase", "parameter_id", "t_seconds", "value"],
            )
            writer.writeheader()
            writer.writerow({
                "patient_id": "patient_001",
                "phase": "Post-operative",
                "parameter_id": "postop_creatinine_72h",
                "t_seconds": "259200",
                "value": "45",
            })

        with patch(
            "src.time_series_plots.write_plotly_html",
            wraps=write_plotly_html,
        ) as write_plot:
            result = TimeSeriesPlotter(self.mapping).generate(
                source,
                self.root / "hours-plot",
                parameter_ids=["postop_creatinine_72h"],
            )

        self.assertEqual(result["observation_count"], 1)
        self.assertTrue(result["plots"][0].is_file())
        html = result["plots"][0].read_text(encoding="utf-8")
        self.assertIn("Elapsed time (hours)", html)
        figure = write_plot.call_args.args[0]
        self.assertEqual(list(figure.data[0].x), [72.0])
        self.assertEqual(figure.layout.xaxis.title.text, "Elapsed time (hours)")
        self.assertIn("Elapsed: %{x:g} h", figure.data[0].hovertemplate)

    def test_class_grouping_creates_one_multi_panel_plot_per_class(self):
        result = TimeSeriesPlotter(self.mapping).generate(
            self.input, self.root / "all-classes", group_by="class",
        )

        self.assertEqual(result["plot_count"], 2)
        self.assertEqual(result["parameter_count"], 3)
        self.assertEqual(result["patient_count"], 2)
        self.assertEqual(result["observation_count"], 7)
        self.assertEqual({path.name for path in result["plots"]}, {
            "epic.LAB.html", "epic.FLUIDS_OUT.html",
        })
        self.assertTrue(all(path.is_file() for path in result["plots"]))

    def test_class_parameter_and_single_patient_selection(self):
        output = self.root / "class-plots"
        result = TimeSeriesPlotter(self.mapping).generate(
            self.input,
            output,
            class_names=["epic.LAB"],
            parameter_ids=["during_creatinine"],
            patient_ids=["patient_002"],
            group_by="class",
        )

        self.assertEqual(result["plot_count"], 1)
        self.assertEqual(result["parameter_count"], 1)
        self.assertEqual(result["patient_count"], 1)
        self.assertEqual(result["observation_count"], 2)
        self.assertEqual(result["plots"][0].name, "epic.LAB.html")

    def test_cli_plots_selected_patient_series(self):
        output = self.root / "cli-plots"
        result = main([
            "plot-time-series",
            "--input", str(self.input),
            "--parameter-map", str(ROOT / "config" / "pipeline.template.xml"),
            "--output", str(output),
            "--group-by", "class",
            "--class", "epic.LAB",
            "--parameter", "during_creatinine",
            "--patient", "patient_002",
            "--log-dir", str(self.root / "logs"),
        ])

        self.assertEqual(result, 0)
        self.assertTrue((output / "epic.LAB.html").is_file())

    def test_cli_shows_safe_plot_dependency_diagnostic(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), patch(
            "src.cli.TimeSeriesPlotter.generate",
            side_effect=RuntimeError(
                "Time-series plots require Plotly; install dependencies from requirements.txt"
            ),
        ):
            result = main([
                "plot-time-series",
                "--input", str(self.input),
                "--parameter-map", str(ROOT / "config" / "pipeline.template.xml"),
                "--output", str(self.root / "error-plots"),
                "--log-dir", str(self.root / "logs"),
            ])

        self.assertEqual(result, 2)
        self.assertIn("Time-series plots require Plotly", stderr.getvalue())

    def test_missing_plotly_dependency_reports_install_path_without_output(self):
        output = self.root / "missing-plotly"
        original_import = __import__

        def block_plotly_import(name, *args, **kwargs):
            if name.startswith("plotly"):
                raise ImportError("Plotly unavailable in this interpreter")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_plotly_import):
            with self.assertRaisesRegex(RuntimeError, "Time-series plots require Plotly"):
                TimeSeriesPlotter(self.mapping).generate(self.input, output)

        self.assertFalse(output.exists())

    def test_unknown_patient_selection_fails_before_creating_output(self):
        output = self.root / "invalid-plots"

        with self.assertRaisesRegex(ValueError, "patient"):
            TimeSeriesPlotter(self.mapping).generate(
                self.input, output, patient_ids=["patient_999"],
            )

        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
