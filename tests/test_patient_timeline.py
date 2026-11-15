"""Synthetic tests for the patient timeline dashboard."""

import csv
from datetime import datetime
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from zoneinfo import ZoneInfo
from unittest.mock import patch

from src.pipeline_runner import run_pipeline

from src.pipeline_config import load_pipeline_config
from src.plotly_output import write_plotly_html
from src.patient_timeline import PatientTimelineDashboard

ROOT = Path(__file__).resolve().parents[1]


class PatientTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mapping = load_pipeline_config(ROOT / "config/pipeline.template.xml")["preprocessing"]["parameter_map"]
        selected_ids = {
            "age", "during_creatinine", "postop_creatinine_72h",
            "mortality_30d", "recorded_death_days",
        }
        self.mapping["parameters"] = [
            parameter for parameter in self.mapping["parameters"]
            if parameter["id"] in selected_ids
        ]
        self.mapping["postoperative_measurements"] = [
            measurement for measurement in self.mapping["postoperative_measurements"]
            if measurement["id"] in {"postop_creatinine_72h", "mortality_30d", "recorded_death_days"}
        ]
        self.input = self.root / "patient.csv"
        with self.input.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "unixtime",
                "epic.EVENTS.EVENT",
                "epic.DEMOGRAPHICS.age",
                "epic.LAB.Kreatinine (umol/L)",
            ])
            writer.writerows([
                [1_700_000_000, "", "8", "45"],
                [1_700_003_600, "Pre-op assessment", "9", ""],
                [1_700_007_200, "Start surgery", "", "40"],
                [1_700_010_800, "Incision", "", "50"],
                [1_700_014_400, "Stop anesthesie", "", "60"],
                [1_700_018_000, "ICU arrival", "", "70"],
            ])

    def tearDown(self):
        self.temporary.cleanup()

    def test_series_dropdown_exposes_every_plotted_field_and_independent_checkboxes(self):
        with self.input.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        rows[0].append("epic.LAB.Glucose (mmol/L)")
        for index, row in enumerate(rows[1:]):
            row.append("5" if index in (2, 3) else "")
        with self.input.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        output = self.root / "series-menu.html"
        with patch("src.patient_timeline.write_plotly_html", wraps=write_plotly_html) as writer:
            PatientTimelineDashboard(self.mapping).generate([self.input], output)
        figure = writer.call_args.args[0]
        self.assertEqual(len(figure.data), 2)
        self.assertIs(figure.layout.showlegend, False)
        self.assertEqual({trace.meta["series"] for trace in figure.data}, {
            "epic.LAB.Glucose (mmol/L)", "epic.LAB.Kreatinine (umol/L)",
        })
        page = output.read_text(encoding="utf-8")
        self.assertIn('id="timeline-series-menu"', page)
        self.assertIn('id="timeline-series-all"', page)
        self.assertIn('id="timeline-series-options"', page)
        self.assertIn("Plotly.restyle", page)

    def test_all_phase_events_and_boundaries_share_one_timeline_without_series_markers(self):
        output = self.root / "phase-events.html"
        with patch("src.patient_timeline.write_plotly_html", wraps=write_plotly_html) as writer:
            PatientTimelineDashboard(self.mapping).generate([self.input], output)
        self.assertFalse(any(trace.name == "Timeline events" for trace in writer.call_args.args[0].data))
        page = output.read_text(encoding="utf-8")
        self.assertIn('id="timeline-events"', page)
        self.assertIn('id="timeline-event-details" class="timeline-card" hidden', page)
        self.assertLess(page.index('id="timeline-events"'), page.index('id="timeline-event-details"'))
        self.assertIn('"timeline_events":[', page)
        for phase, value in (("Pre-operative", "Pre-op assessment"),
                             ("During-operative", "Incision"), ("Post-operative", "ICU arrival")):
            self.assertIn(f'"phase":"{phase}"', page)
            self.assertIn(f'"value":"{value}"', page)
        self.assertIn('"name":"Operation end","hours":2.0', page)

    def test_large_chart_has_toolbar_patient_selector_and_boundary_timeline(self):
        output = self.root / "timeline-layout.html"
        with patch("src.patient_timeline.write_plotly_html", wraps=write_plotly_html) as writer:
            PatientTimelineDashboard(self.mapping).generate([self.input], output, compare_all_patients=True)
        figure = writer.call_args.args[0]
        self.assertGreaterEqual(figure.layout.height, 700)
        self.assertFalse(figure.layout.updatemenus)
        html = output.read_text(encoding="utf-8")
        self.assertLess(html.index('id="timeline-patient"'), html.index('class="plotly-graph-div"'))
        self.assertIn('id="timeline-milestones"', html)
        self.assertIn('"name":"Operation start","hours":0.0', html)
        self.assertIn('"name":"Operation end","hours":2.0', html)
        self.assertIn("Plotly.restyle", html)

    def test_untimed_demographics_render_below_chart_without_inventing_event_time(self):
        with self.input.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        rows[1][2] = rows[2][2] = ""
        rows.append(["", "", "9", ""])
        with self.input.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        output = self.root / "untimed.html"
        with patch("src.patient_timeline.write_plotly_html", wraps=write_plotly_html) as writer:
            PatientTimelineDashboard(self.mapping).generate([self.input], output)
        events = PatientTimelineDashboard(self.mapping)._patient_dashboard_data("patient_001", (self.input,))["timeline_events"]
        for event in events:
            self.assertFalse(any(item["parameter_id"] == "age" for item in event["details"]))
        html = output.read_text(encoding="utf-8")
        self.assertIn('id="timeline-demographics"', html)
        self.assertIn("No timestamp", html)
        self.assertIn('"demographics":[["Age","9"', html)
        self.assertIn("unavailable demographic fields", html)
        self.assertNotIn("1700000000", html)

    def test_dashboard_is_compact_and_tables_are_rendered_on_demand(self):
        output = self.root / "compact.html"
        with patch("src.patient_timeline.write_plotly_html", wraps=write_plotly_html) as write_plot:
            PatientTimelineDashboard(self.mapping).generate([self.input], output)
        figure = write_plot.call_args.args[0]
        self.assertLessEqual(figure.layout.height, 900)
        self.assertGreaterEqual(figure.layout.height, 700)
        self.assertFalse(any(trace.type == "table" for trace in figure.data))
        page = output.read_text(encoding="utf-8")
        for panel in ("timeline-summary", "timeline-pre", "timeline-post", "timeline-event-details"):
            self.assertIn(f'id="{panel}"', page)
        self.assertIn("table-scroll", page)
        self.assertIn("renderPatient", page)
        self.assertIn("Plotly.restyle", page)
        self.assertIn('id="timeline-patient"', page)
        self.assertIn("document.readyState", page)
        self.assertNotIn('"cohort":{"pre"', page)

    def test_dashboard_shows_phase_events_data_and_summary_by_patient(self):
        output = self.root / "patient-timeline.html"
        with patch(
            "src.patient_timeline.write_plotly_html",
            wraps=write_plotly_html,
        ) as write_plot:
            result = PatientTimelineDashboard(self.mapping).generate(
                [self.input], output,
            )

        self.assertEqual(result["patient_count"], 1)
        self.assertEqual(result["event_count"], 5)
        self.assertEqual(result["data_point_count"], 4)
        self.assertGreaterEqual(write_plot.call_args.args[0].layout.height, 700)
        self.assertTrue(output.is_file())
        html = output.read_text(encoding="utf-8")
        self.assertIn("patient_001", html)
        self.assertIn('id="timeline-events"', html)
        self.assertIn("timeline-event-details", html)
        self.assertNotIn("1700000000", html)
        self.assertIn("Operation duration", html)
        self.assertIn("Elapsed hours from operation start", html)
        self.assertIn("Configured during-operative classes", html)
        self.assertIn("94", html)
        self.assertIn("Pre-op assessment", html)
        self.assertIn("Stop anesthesie", html)

        figure = write_plot.call_args.args[0]
        series = figure.data[0]
        self.assertEqual(series.name, "epic.LAB.Kreatinine (umol/L)")
        self.assertEqual(list(series.x), [0.0, 1.0])
        self.assertEqual(list(series.customdata), [40.0, 50.0])
        events = PatientTimelineDashboard(self.mapping)._patient_dashboard_data("patient_001", (self.input,))["timeline_events"]
        self.assertEqual([event["hours"] for event in events], [-1.0, 0.0, 1.0, 2.0, 3.0])
        self.assertEqual([event["phase"] for event in events], [
            "Pre-operative", "During-operative", "During-operative",
            "Post-operative", "Post-operative",
        ])
        selected_events = {item["value"]: item for item in events}
        pre_details = selected_events["Pre-op assessment"]["details"]
        during_details = selected_events["Incision"]["details"]
        post_details = selected_events["ICU arrival"]["details"]
        self.assertFalse(any(item["parameter_id"] == "age" for item in pre_details))
        self.assertIn('"demographics":[["Age","9"', html)
        self.assertTrue(any(item["parameter_id"] == "during_creatinine" and item["value"] == "50"
                            for item in during_details))
        self.assertTrue(any(item["parameter_id"] == "postop_creatinine_72h" and item["value"] == "70"
                            for item in post_details))
        self.assertNotIn("timestamp", str(events).casefold())
        post_script = write_plot.call_args.kwargs["post_script"]
        self.assertIn("showEvent", post_script)
        self.assertIn("timeline-event-details", post_script)
        self.assertNotIn("1700000000", html)
        self.assertFalse(any(trace.type == "table" for trace in figure.data))
        self.assertIn("Pre-op assessment", html)
        self.assertIn("70 umol/L", html)
        self.assertIn("Operation duration", html)
        self.assertIn("Death status", html)
        self.assertIn("Not recorded", html)
        self.assertIn("Distance from operation to death", html)
        self.assertIn("Unavailable", html)

    def test_first_postoperative_end_controls_hospital_duration_and_highlight(self):
        baseline = 1_700_000_000
        source = self.root / "synthetic-stay.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["unixtime", "epic.EVENTS.EVENT"])
            writer.writerows([
                [baseline, "Start surgery"],
                [baseline + 7_200, "Stop anesthesie"],
                [baseline + 1_046_700, "admission end"],
                [baseline + 26_636_472, "Getekend bezoek"],
            ])
        mapping = load_pipeline_config(ROOT / "config/pipeline.template.xml")["preprocessing"]["parameter_map"]
        dashboard = PatientTimelineDashboard(mapping)
        patient = dashboard._patient_dashboard_data("patient_001", (source,))
        self.assertEqual(patient["milestones"][2]["hours"], 290.75)
        self.assertIn(("Time in hospital after operation", "12.03 days", "XML-configured duration; source datetimes are not shown"),
                      patient["summary_rows"])
        self.assertIn(("Time in hospital after operation", "12.03 days"), patient["duration_stats"])
        output = self.root / "synthetic-stay.html"
        dashboard.generate([source], output)
        html = output.read_text(encoding="utf-8")
        self.assertIn('"duration_stats":', html)
        self.assertIn('["Time in hospital after operation","12.03 days"]', html)
        self.assertLess(html.index('id="timeline-duration-stats"'), html.index('id="timeline-demographics"'))
        self.assertIn('Confirm event meaning before interpreting stay durations', html)
        self.assertIn('After post-operative end', html)
        self.assertIn("events.length === 1 ? 'event' : 'events'", html)
        self.assertNotIn('"hours":7399.02', html.split('"milestones":', 1)[1].split('"timeline_events":', 1)[0])

    def test_summary_uses_configured_ic_and_postoperative_hospital_durations(self):
        mapping = load_pipeline_config(ROOT / "config/pipeline.template.xml")["preprocessing"]["parameter_map"]
        selected_ids = {
            "age", "during_creatinine", "postop_creatinine_72h", "icu_los", "hospital_los",
        }
        mapping["parameters"] = [item for item in mapping["parameters"] if item["id"] in selected_ids]
        mapping["postoperative_measurements"] = [
            item for item in mapping["postoperative_measurements"]
            if item["id"] in {"postop_creatinine_72h", "icu_los", "hospital_los"}
        ]
        baseline = 1_700_000_000
        icu_zone = ZoneInfo("Europe/Amsterdam")
        icu_start = datetime.fromtimestamp(baseline + 14_400 + 3_600, icu_zone).replace(tzinfo=None)
        icu_end = datetime.fromtimestamp(baseline + 14_400 + 3_600 + 86_400, icu_zone).replace(tzinfo=None)
        source = self.root / "configured-durations.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "unixtime", "epic.EVENTS.EVENT", "epic.DEMOGRAPHICS.age",
                "epic.LAB.Kreatinine (umol/L)",
                "q_umcg_aki_marco.adm_ic_admdate", "q_umcg_aki_marco.adm_ic_disdate",
            ])
            writer.writerows([
                [baseline, "Pre-op assessment", "9", "", "", ""],
                [baseline + 7_200, "Start surgery", "", "40", "", ""],
                [baseline + 14_400, "Stop anesthesie", "", "60", "", ""],
                [baseline + 18_000, "ICU arrival", "", "70", icu_start.isoformat(), icu_end.isoformat()],
                [baseline + 14_400 + 172_800, "Hospital discharge", "", "", "", ""],
            ])
        output = self.root / "configured-durations.html"

        with patch("src.patient_timeline.write_plotly_html", wraps=write_plotly_html) as write_plot:
            PatientTimelineDashboard(mapping).generate([source], output)

        summary = output.read_text(encoding="utf-8")
        self.assertIn("Time in IC", summary)
        self.assertIn("1.00 days", summary)
        self.assertIn("Time in hospital after operation", summary)
        self.assertIn("2.00 days", summary)
        dashboard_html = output.read_text(encoding="utf-8")
        self.assertNotIn(icu_start.isoformat(), dashboard_html)
        self.assertNotIn(icu_end.isoformat(), dashboard_html)

    def test_death_days_on_a_preoperative_row_appear_in_operation_summary(self):
        source = self.root / "death-days.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "unixtime", "epic.EVENTS.EVENT", "epic.DEMOGRAPHICS.age",
                "epic.LAB.Kreatinine (umol/L)", "epic.DEMOGRAPHICS.days mortality",
            ])
            writer.writerows([
                [1_700_000_000, "Pre-op assessment", "9", "", "4.5"],
                [1_700_007_200, "Start surgery", "", "40", ""],
                [1_700_014_400, "Stop anesthesie", "", "60", ""],
            ])
        output = self.root / "death-days.html"

        with patch("src.patient_timeline.write_plotly_html", wraps=write_plotly_html) as write_plot:
            PatientTimelineDashboard(self.mapping).generate([source], output)

        summary_text = output.read_text(encoding="utf-8")
        self.assertIn("Yes (recorded)", summary_text)
        self.assertIn("Distance from operation to death", summary_text)
        self.assertIn("4.5 days", summary_text)
        dashboard_html = output.read_text(encoding="utf-8")
        self.assertIn("Death status", dashboard_html)
        self.assertIn("4.5 days", dashboard_html)

    def test_zero_days_is_a_recorded_death_at_zero_distance(self):
        source = self.root / "death-day-zero.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "unixtime", "epic.EVENTS.EVENT", "epic.DEMOGRAPHICS.age",
                "epic.LAB.Kreatinine (umol/L)", "epic.DEMOGRAPHICS.days mortality",
            ])
            writer.writerow([1_700_000_000, "Pre-op assessment", "9", "", "0"])
            writer.writerow([1_700_007_200, "Start surgery", "", "40", ""])
            writer.writerow([1_700_014_400, "Stop anesthesie", "", "60", ""])
        output = self.root / "death-day-zero.html"

        with patch("src.patient_timeline.write_plotly_html", wraps=write_plotly_html) as write_plot:
            PatientTimelineDashboard(self.mapping).generate([source], output)

        summary_text = output.read_text(encoding="utf-8")
        self.assertIn("Yes (recorded)", summary_text)
        self.assertIn("0 days", summary_text)

    def test_mortality_flag_shows_death_without_inventing_day_distance(self):
        source = self.root / "death-flag.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "unixtime", "epic.EVENTS.EVENT", "epic.DEMOGRAPHICS.age",
                "epic.LAB.Kreatinine (umol/L)", "q_umcg_aki_marco.adm_deceased",
            ])
            writer.writerow([1_700_000_000, "Pre-op assessment", "9", "", ""])
            writer.writerow([1_700_007_200, "Start surgery", "", "40", ""])
            writer.writerow([1_700_014_400, "Stop anesthesie", "", "60", ""])
            writer.writerow([1_700_018_000, "ICU arrival", "", "70", "1"])
        output = self.root / "death-flag.html"

        with patch("src.patient_timeline.write_plotly_html", wraps=write_plotly_html) as write_plot:
            PatientTimelineDashboard(self.mapping).generate([source], output)

        summary_text = output.read_text(encoding="utf-8")
        self.assertIn("Yes (recorded)", summary_text)
        self.assertIn("Distance from operation to death", summary_text)
        self.assertIn("Unavailable", summary_text)

    def test_dashboard_dropdown_selects_each_pseudonymous_patient(self):
        second_patient = self.root / "second.csv"
        second_patient.write_text(self.input.read_text(encoding="utf-8"), encoding="utf-8")
        output = self.root / "multiple-patients.html"
        with patch(
            "src.patient_timeline.write_plotly_html",
            wraps=write_plotly_html,
        ) as write_plot:
            result = PatientTimelineDashboard(self.mapping).generate(
                [self.input, second_patient], output,
                compare_all_patients=True, theme="dark",
            )

        self.assertEqual(result["patient_count"], 2)
        figure = write_plot.call_args.args[0]
        self.assertFalse(figure.layout.updatemenus)
        self.assertEqual(len(figure.data), 2)
        self.assertEqual({trace.meta["patient"] for trace in figure.data}, {"patient_001", "patient_002"})
        self.assertEqual(figure.layout.template.layout.paper_bgcolor, "rgb(17,17,17)")
        html = output.read_text(encoding="utf-8")
        self.assertNotIn('"visibility":', html)
        self.assertIn('id="timeline-series-menu"', html)
        self.assertIn('id="timeline-patient"', html)
        self.assertIn('"headers":["Patient"', html)
        self.assertIn('"patient_001"', html)
        self.assertIn('"patient_002"', html)

    def test_patient_filter_keeps_requested_patient_pseudonym(self):
        second_patient = self.root / "second.csv"
        second_patient.write_text(self.input.read_text(encoding="utf-8"), encoding="utf-8")
        output = self.root / "selected-patient.html"

        result = PatientTimelineDashboard(self.mapping).generate(
            [self.input, second_patient], output, patient_ids=["patient_002"],
        )

        self.assertEqual(result["patient_count"], 1)
        html = output.read_text(encoding="utf-8")
        self.assertIn("patient_002", html)
        self.assertNotIn("patient_001", html)

    def test_dashboard_can_embed_related_study_plots(self):
        related = self.root / "time_series_plots" / "creatinine.html"
        related.parent.mkdir(parents=True)
        related.write_text("<html>synthetic plot</html>", encoding="utf-8")
        output = self.root / "patient-timeline.html"

        with patch("src.patient_timeline.write_plotly_html", wraps=write_plotly_html) as write_plot:
            PatientTimelineDashboard(self.mapping).generate(
                [self.input], output, related_plots=[related],
            )

        post_script = write_plot.call_args.kwargs["post_script"]
        self.assertIn("study-plots", post_script)
        self.assertIn("creatinine.html", post_script)
        self.assertIn("createElement('iframe')", post_script)
        self.assertIn("time_series_plots/creatinine.html", post_script)

    def test_static_preoperative_summary_omits_null_unit_text(self):
        mapping = dict(self.mapping)
        age = next(parameter for parameter in self.mapping["parameters"] if parameter["id"] == "age")
        mapping["parameters"] = [{**age, "allow_static_cutoff_fallback": True}]
        source = self.root / "static.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["unixtime", "epic.EVENTS.EVENT", "epic.DEMOGRAPHICS.age"])
            writer.writerow([1_700_000_000, "", "9"])
        output = self.root / "static-summary.html"

        PatientTimelineDashboard(mapping).generate([source], output)

        html = output.read_text(encoding="utf-8")
        self.assertNotIn("9 None", html)
        self.assertIn("Age", html)
        self.assertIn("No timestamp", html)
        self.assertIn('"demographics":[["Age","9"', html)

    def test_unplaced_records_are_reported_without_fake_zero_times(self):
        with self.input.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["", "Untimed assessment", "10", ""])
        output = self.root / "untimed.html"

        result = PatientTimelineDashboard(self.mapping).generate([self.input], output)

        self.assertEqual(result["patient_count"], 1)
        html = output.read_text(encoding="utf-8")
        self.assertIn("1 event(s), 0 data value(s)", html)

    def test_xml_runner_writes_configured_patient_timeline_dashboard(self):
        input_dir = self.root / "input"
        input_dir.mkdir()
        source = input_dir / "family_001.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "unixtime",
                "epic.EVENTS.EVENT",
                "epic.DEMOGRAPHICS.age",
                "epic.LAB.Kreatinine (umol/L)",
            ])
            writer.writerows([
                [1_700_000_000, "Pre-op assessment", "9", ""],
                [1_700_003_600, "Start surgery", "", "40"],
                [1_700_007_200, "Stop anesthesie", "", "50"],
                [1_700_011_200, "ICU arrival", "", "60"],
            ])

        root = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
        root.find("logging/directory").text = "logs"
        preprocessing = root.find("preprocessing")
        preprocessing.find("run").text = "true"
        inputs = preprocessing.find("inputs")
        for child in list(inputs):
            inputs.remove(child)
        ET.SubElement(inputs, "path").text = str(input_dir)
        preprocessing.find("outputDirectory").text = "results"
        plotting = preprocessing.find("plotting")
        plotting.find("run").text = "false"
        timeline_config = plotting.find("patientTimeline")
        timeline_config.find("run").text = "true"
        timeline_config.find("output").text = "results/patient_timeline_dashboard.html"
        output = self.root / "results" / "patient_timeline_dashboard.html"
        study_plot = self.root / "results" / "parameter_distributions" / "synthetic-study.html"
        study_plot.parent.mkdir(parents=True, exist_ok=True)
        study_plot.write_text("<html>synthetic study chart</html>", encoding="utf-8")
        config_path = self.root / "config" / "pipeline.xml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(config_path, encoding="utf-8", xml_declaration=True)

        result = run_pipeline(config_path)

        self.assertIn("patient timeline dashboard", result)
        self.assertTrue(output.is_file())
        html = output.read_text(encoding="utf-8")
        self.assertIn("parameter_distributions/synthetic-study.html", html)
        self.assertIn("patient_001", html)
        self.assertIn("Pre-op assessment", html)
        self.assertIn("ICU arrival", html)
        self.assertNotIn("patient_002", html)


if __name__ == "__main__":
    unittest.main()
