import csv
import json
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from src.cli import main
from src.models.novel_data.config import NovelDataConfig
from src.models.perceptron.config import PerceptronModelConfig
from src.models.perceptron.perceptron import predict_model, train_model
from src.models.plotting.config import PlotConfig
from src.models.training_validation.config import TrainingValidationConfig
from src.pipeline.config import PipelineConfig, load_pipeline_config
from src.preprocessing.parameters.settings import select_parameter_map
from src.preprocessing.plotting.patient_timeline.timeline_dashboard_view import DashboardLayout

ROOT = Path(__file__).resolve().parents[1]


class PipelineConfigurationTests(unittest.TestCase):
    def test_logger_is_a_direct_pipeline_configuration_child(self):
        for path in (ROOT / "config" / "pipeline.xml", ROOT / "config" / "pipeline.template.xml"):
            xml_root = ET.parse(path).getroot()
            self.assertIsNotNone(xml_root.find("logging"))
            self.assertIsNone(xml_root.find("general"))
        config = load_pipeline_config(ROOT / "config" / "pipeline.xml")
        self.assertEqual(config["logging"]["level"], "INFO")

    def test_default_xml_config_controls_each_pipeline_stage(self):
        config = load_pipeline_config(ROOT / "config" / "pipeline.xml")
        self.assertIsInstance(config["preprocessing"]["run"], bool)
        self.assertIsInstance(config["preprocessing"]["parameter_descriptions"]["run"], bool)
        self.assertIsInstance(config["preprocessing"]["csv_header_descriptions"]["run"], bool)
        self.assertEqual(config["preprocessing"]["csv_header_descriptions"]["inventory_output"], ROOT / "results" / "csv_headers.csv")
        self.assertEqual([phase.name for phase in PipelineConfig.load(ROOT / "config" / "pipeline.xml").preprocessing.parameters.phases], [
            "Pre-operative", "During-operative", "Post-operative"
        ])
        self.assertEqual(config["model"]["active_model"], "perceptron")
        active_model = config["model"]["active_config"]
        self.assertIsInstance(active_model, PerceptronModelConfig)
        self.assertIsInstance(active_model.training_validation, TrainingValidationConfig)
        self.assertIsInstance(active_model.novel_data, NovelDataConfig)
        self.assertIsInstance(active_model.plotting, PlotConfig)
        self.assertEqual(active_model.training_validation.omit, ["operation_diagnosis_proxy"])
        self.assertIsInstance(config["model"]["models"]["perceptron"]["run"], bool)
        self.assertIn(config["model"]["models"]["perceptron"]["mode"], {"train-validation", "novel-data"})
        self.assertEqual(
            config["model"]["models"]["perceptron"]["training_validation"]["omit"],
            ["operation_diagnosis_proxy"],
        )
        self.assertEqual(config["logging"]["level"], "INFO")
        parameters = config["preprocessing"]["parameter_map"]["parameters"]
        self.assertEqual(len(parameters), 20)
        self.assertEqual(sum(item["phase"] == "Pre-operative" for item in parameters), 9)
        self.assertEqual(sum(item["phase"] == "During-operative" for item in parameters), 1)
        self.assertEqual(sum(item["phase"] == "Post-operative" for item in parameters), 10)
        self.assertEqual(config["preprocessing"]["parameter_map"]["included_phases"], [
            "Pre-operative", "During-operative", "Post-operative"
        ])

    def test_active_xml_configures_patient_timeline_dashboard_by_default(self):
        config = load_pipeline_config(ROOT / "config" / "pipeline.xml")
        dashboard = config["preprocessing"]["plotting"].get("patient_timeline")

        self.assertIsNotNone(dashboard)
        self.assertTrue(dashboard["run"])
        self.assertEqual(dashboard["patient_ids"], [])
        self.assertFalse(dashboard["compare_all_patients"])
        self.assertTrue(dashboard["include_study_plots"])
        self.assertEqual(config["preprocessing"]["plotting"]["theme"], "dark")
        self.assertEqual(
            dashboard["output"],
            ROOT / "results" / "patient_timeline_dashboard.html",
        )

    def test_patient_timeline_layout_is_typed_and_configurable_in_both_xml_files(self):
        for name in ("pipeline.xml", "pipeline.template.xml"):
            with self.subTest(name=name):
                root = ET.parse(ROOT / "config" / name).getroot()
                layout = root.find("preprocessing/plotting/patientTimeline/layout")
                self.assertIsNotNone(layout)
                loaded = load_pipeline_config(ROOT / "config" / name)["preprocessing"]["plotting"]["patient_timeline"]["layout"]
                self.assertIsInstance(loaded, DashboardLayout)
                self.assertEqual((loaded.chart_height, loaded.panel_max_height), (740, 320))
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config/pipeline.template.xml").getroot()
            layout = root.find("preprocessing/plotting/patientTimeline/layout")
            layout.find("chartHeight").text = "640"
            layout.find("panelMaxHeight").text = "240"
            path = Path(temporary) / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            loaded = load_pipeline_config(path)["preprocessing"]["plotting"]["patient_timeline"]["layout"]
            self.assertEqual((loaded.chart_height, loaded.panel_max_height), (640, 240))
            layout.find("chartHeight").text = "0"
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "chartHeight"):
                load_pipeline_config(path)

    def test_patient_timeline_patient_ids_are_loaded_from_xml(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
            timeline = root.find("preprocessing/plotting/patientTimeline")
            ET.SubElement(timeline.find("patientIds"), "id").text = "patient_002"
            path = Path(temporary) / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

            config = load_pipeline_config(path)

        self.assertEqual(
            config["preprocessing"]["plotting"]["patient_timeline"]["patient_ids"],
            ["patient_002"],
        )

    def test_template_disables_all_execution_switches(self):
        config = load_pipeline_config(ROOT / "config" / "pipeline.template.xml")
        self.assertFalse(config["preprocessing"]["run"])
        self.assertFalse(config["preprocessing"]["parameter_descriptions"]["run"])
        self.assertFalse(config["preprocessing"]["csv_header_descriptions"]["run"])
        self.assertFalse(config["preprocessing"]["plotting"]["run"])
        self.assertFalse(config["preprocessing"]["plotting"]["time_series"]["run"])
        self.assertFalse(config["preprocessing"]["plotting"]["missingness_by_patient"]["run"])
        self.assertFalse(config["preprocessing"]["plotting"]["patient_timeline"]["run"])
        self.assertFalse(config["model"]["models"]["perceptron"]["run"])

    def test_active_xml_configures_patient_missingness_plot_by_default(self):
        config = load_pipeline_config(ROOT / "config" / "pipeline.xml")
        plotting = config["preprocessing"]["plotting"]["missingness_by_patient"]

        self.assertTrue(plotting["run"])
        self.assertEqual(
            plotting["output"],
            ROOT / "results" / "preoperative_missingness_heatmap.html",
        )

    def test_active_xml_configures_time_series_plotting_for_all_data(self):
        config = load_pipeline_config(ROOT / "config" / "pipeline.xml")
        plotting = config["preprocessing"]["plotting"]["time_series"]

        self.assertTrue(plotting["run"])
        self.assertEqual(plotting["group_by"], "parameter")
        self.assertEqual(plotting["parameter_ids"], [])
        self.assertEqual(plotting["class_names"], [])
        self.assertEqual(plotting["patient_ids"], [])
        self.assertEqual(plotting["output_directory"], ROOT / "results" / "time_series_plots")

    def test_time_series_plot_filters_are_loaded_from_xml_items(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
            time_series = root.find("preprocessing/plotting/timeSeries")
            time_series.find("run").text = "true"
            time_series.find("groupBy").text = "class"
            ET.SubElement(time_series.find("parameterIds"), "id").text = "during_creatinine"
            ET.SubElement(time_series.find("classNames"), "name").text = "epic.LAB"
            ET.SubElement(time_series.find("patientIds"), "id").text = "patient_002"
            path = Path(temporary) / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

            config = load_pipeline_config(path)
            plotting = config["preprocessing"]["plotting"]["time_series"]

            self.assertEqual(plotting["group_by"], "class")
            self.assertEqual(plotting["parameter_ids"], ["during_creatinine"])
            self.assertEqual(plotting["class_names"], ["epic.LAB"])
            self.assertEqual(plotting["patient_ids"], ["patient_002"])
            self.assertEqual(plotting["output_directory"], Path(temporary) / "results/time_series_plots")

    def test_time_series_plotting_requires_export_when_preprocessing_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
            root.find("preprocessing/run").text = "true"
            root.find("preprocessing/plotting/timeSeries/run").text = "true"
            path = Path(temporary) / "config" / "pipeline.xml"
            path.parent.mkdir()
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

            with self.assertRaisesRegex(ValueError, "export_time_series=true"):
                load_pipeline_config(path)

    def test_unknown_active_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "pipeline.xml"
            text = (ROOT / "config" / "pipeline.xml").read_text(encoding="utf-8")
            config_path.write_text(
                text.replace("<activeModel>perceptron</activeModel>", "<activeModel>randomForest</activeModel>"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported model type: randomForest"):
                load_pipeline_config(config_path)

    def test_parameter_selection_filters_both_parameters_and_postoperative_measurements(self):
        parameter_map = {
            "parameters": [
                {"id": "age", "phase": "Pre-operative"},
                {"id": "sex", "phase": "Pre-operative"},
                {"id": "aki", "phase": "Post-operative"},
            ],
            "postoperative_measurements": [
                {"id": "aki"},
                {"id": "los"},
            ],
        }
        selected = select_parameter_map(
            parameter_map,
            phases=["Pre-operative", "Post-operative"],
            include_ids=["age", "aki"],
            exclude_ids=[],
        )
        self.assertEqual([item["id"] for item in selected["parameters"]], ["age", "aki"])
        self.assertEqual([item["id"] for item in selected["postoperative_measurements"]], ["aki"])

    def test_selection_rejects_unknown_parameter_ids(self):
        with self.assertRaisesRegex(ValueError, "Unknown parameter ID"):
            select_parameter_map(
                {"parameters": [{"id": "age", "phase": "Pre-operative"}]},
                phases=["Pre-operative"],
                include_ids=["missing"],
                exclude_ids=[],
            )

    def test_xml_runner_trains_then_predicts_novel_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "config").mkdir()
            (project / "data" / "patients").mkdir(parents=True)
            shutil.copy(ROOT / "config" / "pipeline.xml", project / "config" / "pipeline.xml")
            from tests.test_patient_vectors import PatientVectorTests
            patient_writer = PatientVectorTests()
            for index, creatinine in enumerate(("100", "100", "100", "45", "45")):
                patient_writer.make_patient(
                    project / "data" / "patients",
                    f"synthetic-{index}.csv",
                    postop_creatinine=creatinine,
                    operation_result="severe problems" if index < 3 else "optimal",
                )

            config_path = project / "config" / "pipeline.xml"
            tree = ET.parse(config_path)
            root = tree.getroot()
            root.find("logging/directory").text = "logs"
            root.find("preprocessing/run").text = "true"
            root.find("preprocessing/inputDirectory").text = "data/patients"
            root.find("preprocessing/outputDirectory").text = "results"
            root.find("preprocessing/parameterDescriptions/run").text = "true"
            root.find("preprocessing/parameterDescriptions/output").text = "results/parameter_reference.md"
            root.find("preprocessing/plotting/run").text = "false"
            root.find("preprocessing/plotting/distributions").text = "false"
            root.find("model/perceptron/run").text = "true"
            root.find("model/perceptron/mode").text = "train-validation"
            root.find("model/perceptron/trainingValidation/trainingSet").text = "results/training_set.csv"
            root.find("model/perceptron/trainingValidation/validationSet").text = "results/validation_set.csv"
            root.find("model/perceptron/trainingValidation/modelOutput").text = "results/perceptron_model.json"
            root.find("model/perceptron/novelData/modelInput").text = "results/perceptron_model.json"
            root.find("model/perceptron/novelData/predictions").text = "results/novel_predictions.csv"
            root.find("model/perceptron/plotting/run").text = "true"
            root.find("model/perceptron/plotting/output").text = "results/perceptron_confusion_matrix.html"
            ET.indent(root, space="  ")
            tree.write(config_path, encoding="utf-8", xml_declaration=True)
            self.assertEqual(
                load_pipeline_config(config_path)["preprocessing"]["input_directory"],
                (project / "data" / "patients").resolve(),
            )

            self.assertEqual(main(["run", "--config", str(config_path)]), 0)
            results = project / "results"
            self.assertTrue((results / "patient_vectors.csv").is_file())
            self.assertTrue((results / "patient_outcomes.csv").is_file())
            self.assertTrue((results / "parameter_reference.md").is_file())
            model_path = results / "perceptron_model.json"
            self.assertTrue(model_path.is_file())
            self.assertEqual(
                json.loads(model_path.read_text(encoding="utf-8"))["omitted_parameters"],
                ["operation_diagnosis_proxy"],
            )
            self.assertTrue((results / "perceptron_confusion_matrix.html").is_file())
            dashboard_html = (results / "patient_timeline_dashboard.html").read_text(encoding="utf-8")
            self.assertIn("perceptron_confusion_matrix.html", dashboard_html)
            self.assertIn("time_series_plots/", dashboard_html)

            root.find("model/perceptron/mode").text = "novel-data"
            root.find("model/perceptron/plotting/run").text = "false"
            ET.indent(root, space="  ")
            tree.write(config_path, encoding="utf-8", xml_declaration=True)
            self.assertEqual(main(["run", "--config", str(config_path)]), 0)
            with (results / "novel_predictions.csv").open(newline="", encoding="utf-8") as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(len(predictions), 5)
            self.assertIn("prediction", predictions[0])
            self.assertNotIn("target", predictions[0])

    def test_saved_perceptron_predicts_novel_rows_without_targets(self):
        parameters = [{"id": "measurement", "parameter_type": "numeric"}]
        rows = [
            {"patient_id": f"p{i}", "measurement": str(i), "target": -1 if i < 5 else 1}
            for i in range(10)
        ]
        model = train_model(rows, parameters, train_fraction=0.8, epochs=20, seed=3)
        predictions = predict_model(
            model,
            [{"patient_id": "new-patient", "measurement": "0"}],
        )
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["patient_id"], "new-patient")
        self.assertEqual(predictions[0]["prediction"], -1)
        self.assertNotIn("target", predictions[0])


if __name__ == "__main__":
    unittest.main()
