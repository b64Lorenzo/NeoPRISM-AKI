"""Shared Plotly artifact output tests."""

import tempfile
import unittest
from pathlib import Path

import plotly.graph_objects as go

from src.plotly_output import plotly_template, write_plotly_html


class PlotlyOutputTests(unittest.TestCase):
    def test_theme_setting_resolves_only_supported_plotly_templates(self):
        self.assertEqual(plotly_template("light"), "plotly_white")
        self.assertEqual(plotly_template("dark"), "plotly_dark")
        with self.assertRaisesRegex(ValueError, "light or dark"):
            plotly_template("custom")

    def test_writes_offline_html_and_reuses_one_local_plotly_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "plots"
            figure = go.Figure(go.Scatter(x=[0, 1], y=[1, 2], mode="lines"))

            first = write_plotly_html(figure, output / "first.html")
            second = write_plotly_html(figure, output / "second.html")

            self.assertEqual(first, output / "first.html")
            self.assertEqual(second, output / "second.html")
            bundle = output / "plotly.min.js"
            self.assertTrue(bundle.is_file())
            self.assertGreater(bundle.stat().st_size, 1_000_000)
            for path in (first, second):
                html = path.read_text(encoding="utf-8")
                self.assertIn("plotly.min.js", html)
                self.assertIn("Plotly.newPlot", html)
                self.assertNotIn("cdn.plot.ly", html)

    def test_injects_optional_post_script_after_plot_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "interactive.html"
            figure = go.Figure(go.Scatter(x=[0], y=[1]))

            write_plotly_html(
                figure, path,
                post_script="document.getElementById('{plot_id}').dataset.ready = 'yes';",
            )

            html = path.read_text(encoding="utf-8")
            self.assertIn("dataset.ready", html)
            self.assertIn("document.getElementById(", html)

    def test_rejects_non_html_output_suffix_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plot.png"
            figure = go.Figure(go.Scatter(x=[0], y=[1]))

            with self.assertRaisesRegex(ValueError, r"\.html"):
                write_plotly_html(figure, path)

            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
