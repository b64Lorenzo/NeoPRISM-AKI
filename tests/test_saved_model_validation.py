"""Saved-model transforms must fail explicitly, never fabricate a prediction."""

import copy
import unittest

from src.models.perceptron.perceptron import predict_model


class SavedModelValidationTests(unittest.TestCase):
    def model(self):
        return {
            "feature_ids": ["age"],
            "feature_names": ["age"],
            "weights": [1.0],
            "numeric_transform": {"age": {"mean": 0.0, "scale": 1.0}},
            "categorical_values": {},
        }

    def test_invalid_numeric_transforms_raise_value_error(self):
        invalid = (
            {"mean": 0, "scale": 0},
            {"mean": 0, "scale": -1},
            {"mean": float("nan"), "scale": 1},
            {"mean": 0, "scale": float("inf")},
            {"mean": True, "scale": 1},
            {"mean": 0},
            None,
        )
        for stats in invalid:
            with self.subTest(stats=stats):
                model = self.model()
                model["numeric_transform"]["age"] = stats
                with self.assertRaisesRegex(ValueError, "Saved model numeric transform"):
                    predict_model(model, [{"patient_id": "p1", "age": "2"}])

    def test_valid_numeric_transform_is_not_mutated(self):
        model = self.model()
        original = copy.deepcopy(model)
        self.assertEqual(
            predict_model(model, [{"patient_id": "p1", "age": "2"}])[0]["prediction"], 1
        )
        self.assertEqual(model, original)

    def test_finite_inputs_cannot_silently_overflow_the_encoder(self):
        model = self.model()
        model["numeric_transform"]["age"]["scale"] = 1e-300
        with self.assertRaisesRegex(ValueError, "Non-finite encoded numeric input"):
            predict_model(model, [{"patient_id": "p1", "age": "1e100"}])
