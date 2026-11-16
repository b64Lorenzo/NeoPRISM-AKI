"""Exercise the shared train/save path and explicit model-runner dispatch."""

import json
import tempfile
import unittest
from pathlib import Path


class ModelWorkflowTests(unittest.TestCase):
    def test_shared_training_workflow_writes_disjoint_outputs_without_holdout_targets(self):
        from src.models.perceptron.runner import train_and_save
        from src.models.training_validation.config import TrainingValidationConfig
        from src.utils.table_io import read_csv_rows, write_csv_rows

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vectors = [{"patient_id": f"p{index}", "age": str(index)} for index in range(12)]
            outcomes = [
                {"patient_id": row["patient_id"], "target": "-1" if index < 6 else "1"}
                for index, row in enumerate(vectors)
            ]
            write_csv_rows(root / "vectors.csv", vectors, ["patient_id", "age"])
            write_csv_rows(
                root / "outcomes.csv", list(reversed(outcomes)), ["patient_id", "target"]
            )
            settings = TrainingValidationConfig(
                vectors=root / "vectors.csv",
                outcomes=root / "outcomes.csv",
                training_set=root / "out" / "training.csv",
                validation_set=root / "out" / "validation.csv",
                model_output=root / "out" / "model.json",
                train_fraction=0.75,
                epochs=20,
                seed=2,
                omit=[],
            )
            model = train_and_save(
                settings,
                {
                    "parameters": [
                        {"id": "age", "phase": "Pre-operative", "parameter_type": "numeric"}
                    ]
                },
            )
            train_fields, training = read_csv_rows(settings.training_set)
            validation_fields, validation = read_csv_rows(settings.validation_set)
            self.assertEqual(train_fields, ["patient_id", "age", "target"])
            self.assertEqual(validation_fields, ["patient_id", "age", "prediction"])
            self.assertEqual(len(training), model["training_size"])
            self.assertEqual(len(validation), model["validation_size"])
            self.assertFalse(
                {row["patient_id"] for row in training} & {row["patient_id"] for row in validation}
            )
            self.assertEqual(len(training) + len(validation), len(vectors))
            self.assertEqual(json.loads(settings.model_output.read_text(encoding="utf-8")), model)
            self.assertNotIn("_training_set", model)
            self.assertNotIn("_validation_set", model)

    def test_unknown_model_runner_is_rejected_before_execution(self):
        from dataclasses import dataclass

        from src.models.base import ModelConfig
        from src.models.runner import get_model_runner

        @dataclass(frozen=True)
        class UnknownConfig(ModelConfig):
            xml_name = "unknown"

            @classmethod
            def from_xml(cls, element, project_root):
                return cls()

        with self.assertRaisesRegex(ValueError, "No runner is registered for model: unknown"):
            get_model_runner(UnknownConfig())
