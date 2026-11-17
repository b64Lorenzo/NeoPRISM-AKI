"""Integration contracts for optional preprocessing and model collections."""

import copy
import csv
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from src.pipeline.config import PipelineConfig
from src.pipeline.runner import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def _find(parent: ET.Element, path: str) -> ET.Element:
    element = parent.find(path)
    if element is None:
        raise AssertionError(f"Missing XML element: {path}")
    return element


class OptionalPipelineModelTests(unittest.TestCase):
    def test_pipeline_without_preprocessing_or_models_is_a_noop(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            config_dir = project / "config"
            config_dir.mkdir()
            config_path = config_dir / "pipeline.xml"
            config_path.write_text(
                """<?xml version='1.0'?>
<pipelineConfiguration version='2'>
  <logging><directory>logs</directory><level>ERROR</level></logging>
</pipelineConfiguration>
""",
                encoding="utf-8",
            )

            config = PipelineConfig.load(config_path)
            self.assertIsNone(config.preprocessing)
            self.assertEqual(config.models.models, ())
            self.assertEqual(run_pipeline(config_path), "no stages enabled")

    def test_multiple_models_use_their_own_clean_inputs_without_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            config_dir = project / "config"
            config_dir.mkdir()
            root = ET.parse(ROOT / "config" / "pipeline.template.xml").getroot()
            root.set("version", "2")
            root.remove(_find(root, "preprocessing"))
            old_model = _find(root, "models/model")
            root.remove(_find(root, "models"))
            models = ET.SubElement(root, "models")

            for model_id, values in (
                ("baseline", ("0", "1", "10", "11")),
                ("alternate", ("20", "21", "30", "31")),
            ):
                model = ET.SubElement(models, "model", id=model_id, type="perceptron")
                for child in old_model:
                    model.append(copy.deepcopy(child))
                _find(model, "run").text = "true"
                _find(model, "plotting/run").text = "false"
                training = _find(model, "trainingValidation")
                _find(training, "vectors").text = f"data/{model_id}/vectors.csv"
                _find(training, "outcomes").text = f"data/{model_id}/outcomes.csv"
                _find(training, "trainingSet").text = f"results/{model_id}/training.csv"
                _find(training, "validationSet").text = f"results/{model_id}/validation.csv"
                _find(training, "modelOutput").text = f"results/{model_id}/model.json"
                _find(training, "omit").clear()
                features = _find(training, "features")
                features.clear()
                ET.SubElement(features, "feature", id="age", type="numeric")

                data_dir = project / "data" / model_id
                data_dir.mkdir(parents=True)
                with (data_dir / "vectors.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["patient_id", "age"])
                    writer.writeheader()
                    writer.writerows(
                        {"patient_id": f"{model_id}-{index}", "age": value}
                        for index, value in enumerate(values)
                    )
                with (data_dir / "outcomes.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["patient_id", "target"])
                    writer.writeheader()
                    writer.writerows(
                        {"patient_id": f"{model_id}-{index}", "target": "-1" if index < 2 else "1"}
                        for index in range(4)
                    )

            config_path = config_dir / "pipeline.xml"
            ET.ElementTree(root).write(config_path, encoding="utf-8", xml_declaration=True)

            self.assertEqual(
                run_pipeline(config_path),
                "baseline: perceptron training/validation, alternate: perceptron training/validation",
            )
            for model_id in ("baseline", "alternate"):
                self.assertTrue((project / "results" / model_id / "model.json").is_file())
                self.assertTrue((project / "results" / model_id / "training.csv").is_file())
                self.assertTrue((project / "results" / model_id / "validation.csv").is_file())


if __name__ == "__main__":
    unittest.main()
