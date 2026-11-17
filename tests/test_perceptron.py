import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models.perceptron.perceptron import plot_confusion_matrix, train_model
from src.plotlib.plotly_output import write_plotly_html


class PerceptronTests(unittest.TestCase):
    def setUp(self):
        self.parameters = [
            {"id": "measurement", "parameter_type": "numeric"},
            {"id": "group", "parameter_type": "categorical"},
        ]
        self.rows = [
            {"measurement": str(value), "group": "A" if value < 5 else "B", "target": "-1" if value < 5 else "1"}
            for value in range(10)
        ]

    def test_confusion_matrix_supports_configured_dark_theme(self):
        model = {"confusion": {
            "failure_predicted_failure": 1,
            "failure_predicted_success": 0,
            "success_predicted_failure": 0,
            "success_predicted_success": 1,
        }}
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "src.models.perceptron.perceptron.write_plotly_html",
                wraps=write_plotly_html,
            ) as write_plot:
                plot_confusion_matrix(model, Path(temporary) / "confusion.html", theme="dark")
        self.assertEqual(
            write_plot.call_args.args[0].layout.template.layout.paper_bgcolor,
            "rgb(17,17,17)",
        )

    def test_training_uses_repeatable_random_holdout_and_reports_generalization_error(self):
        first = train_model(self.rows, self.parameters, train_fraction=0.7, epochs=20, seed=41)
        second = train_model(self.rows, self.parameters, train_fraction=0.7, epochs=20, seed=41)

        self.assertEqual(first, second)
        self.assertEqual(first["training_size"], 7)
        self.assertEqual(first["validation_size"], 3)
        self.assertGreaterEqual(first["generalization_error"], 0)
        self.assertLessEqual(first["generalization_error"], 1)
        self.assertIn("group=A", first["feature_names"])
        self.assertIn("group=B", first["feature_names"])

    def test_only_exact_binary_targets_are_accepted(self):
        rows = [{"measurement": str(value), "target": -1 if value < 5 else 1} for value in range(10)]
        model = train_model(rows, self.parameters[:1], train_fraction=0.7, epochs=20, seed=3)
        self.assertEqual(model["target_values"], [-1, 1])
        rows[0]["target"] = -100
        with self.assertRaisesRegex(ValueError, "exactly -1.*or 1"):
            train_model(rows, self.parameters[:1], train_fraction=0.7, epochs=20, seed=3)

    def test_omitted_parameters_are_not_encoded(self):
        model = train_model(
            self.rows,
            self.parameters,
            omitted_parameters=["group"],
            train_fraction=0.7,
            epochs=10,
            seed=5,
        )

        self.assertEqual(model["omitted_parameters"], ["group"])
        self.assertTrue(all(not name.startswith("group=") for name in model["feature_names"]))

    def test_parameter_omissions_are_supplied_by_the_caller_not_built_into_the_trainer(self):
        parameters = [
            {"id": "measurement", "parameter_type": "numeric"},
            {"id": "operation_diagnosis_proxy", "parameter_type": "categorical"},
        ]
        rows = [
            {
                "measurement": str(value),
                "operation_diagnosis_proxy": "procedure-a" if value < 5 else "procedure-b",
                "target": -1 if value < 5 else 1,
            }
            for value in range(10)
        ]
        model = train_model(rows, parameters, train_fraction=0.7, epochs=10, seed=5)
        self.assertIn("operation_diagnosis_proxy", model["feature_ids"])
        self.assertEqual(model["omitted_parameters"], [])

    def test_unknown_omitted_parameter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown omitted parameter"):
            train_model(self.rows, self.parameters, omitted_parameters=["not_a_feature"])

    def test_join_uses_patient_id_and_rejects_unmatched_outcomes(self):
        from src.utils.table_io import join_patient_vectors as _join_patient_vectors

        vectors = [{"patient_id": "a", "age": "2"}, {"patient_id": "b", "age": "8"}]
        outcomes = [{"patient_id": "b", "target": "1"}, {"patient_id": "a", "target": "-1"}]
        joined = _join_patient_vectors(vectors, outcomes)
        self.assertEqual([row["target"] for row in joined], ["-1", "1"])
        with self.assertRaisesRegex(ValueError, "patient_id sets do not match"):
            _join_patient_vectors(vectors, outcomes[:1])

if __name__ == "__main__":
    unittest.main()
