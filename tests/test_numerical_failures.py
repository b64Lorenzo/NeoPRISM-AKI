import numpy as np
import pytest

from src.models.logistic_regression import LogisticRegression
from src.models.neural_network import MultiLayerPerceptron


@pytest.mark.parametrize("model", [LogisticRegression(), MultiLayerPerceptron([1, 2])])
def test_finite_but_overflowing_training_fails_clearly(model):
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(
        ValueError, match="[Nn]on-finite|overflow|diverg"
    ):
        model.fit(
            [[1e308], [-1e308]], [-1, 1], epochs=1, learning_rate=1.0, batch_size=2
        )
