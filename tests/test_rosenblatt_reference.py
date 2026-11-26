"""Exact course semantics, including arithmetic order and update-count strengths."""

import copy

import pytest

from src.models.rosenblatt_perceptron.core import train


def test_default_is_homogeneous_fixed_order_with_per_pattern_strengths():
    inputs = [[1.0, 0.0], [0.0, 1.0]]
    model, stats = train(inputs, [1, -1], epochs=10)
    assert model.weights == [0.5, -0.5]
    assert model.bias == 0.0
    assert stats["epochs_run"] == 1
    assert stats["updates"] == 2
    assert stats["converged"] is True
    assert stats["strengths"] == [1, 1]
    assert stats["learning_rate"] == 0.5
    assert stats["clamped"] is False


def test_update_threshold_is_not_the_convergence_threshold():
    model, stats = train([[1.0]], [1], epochs=4, threshold=2.0)
    assert model.weights == [1.0]
    assert stats["epochs_run"] == 1
    assert stats["converged"] is True
    assert stats["history"][-1]["violations"] == 1
    assert stats["history"][-1]["nonpositive_margins"] == 0


def test_negative_finite_update_threshold_matches_reference():
    model, stats = train([[1.0]], [1], epochs=3, threshold=-1.0)
    assert model.weights == [0.0]
    assert stats["updates"] == 0
    assert stats["converged"] is False
    assert stats["history"][-1]["violations"] == 0
    assert stats["history"][-1]["nonpositive_margins"] == 1


def test_clamped_mode_matches_explicit_constant_input_without_mutation():
    inputs = [[1.0], [2.0]]
    original = copy.deepcopy(inputs)
    labels = [-1, 1]
    homogeneous, no_bias = train(inputs, labels, epochs=100)
    model, stats = train(inputs, labels, epochs=100, clamped=True)
    explicit, explicit_stats = train([[1.0, 1.0], [2.0, 1.0]], labels, epochs=100)
    assert no_bias["converged"] is False
    assert homogeneous.bias == 0.0
    assert stats["converged"] is True
    assert model.weights + [model.bias] == explicit.weights
    assert stats["learning_rate"] == 0.5
    assert stats["strengths"] == explicit_stats["strengths"]
    assert stats["updates"] == sum(stats["strengths"])
    assert stats["epochs_run"] == explicit_stats["epochs_run"]
    assert inputs == original


def test_reference_tuple_api_has_identical_arguments_and_return_contract():
    from src.models.rosenblatt_perceptron.core import train_rosenblatt

    result = train_rosenblatt([([1.0, 0.0], 1), ([0.0, 1.0], -1)], max_epochs=10)
    assert result == ([0.5, -0.5], 1, 2, True, [1, 1])


@pytest.mark.parametrize("clamped", [False, True])
@pytest.mark.parametrize("threshold", [-0.25, 0.0, 0.5, 2.0])
@pytest.mark.parametrize("dimensions", [1, 3, 7])
def test_entire_training_trace_matches_pinned_course_reference(
    clamped, threshold, dimensions
):
    from random import Random

    from fixtures.nnci_rosenblatt_reference import train_rosenblatt as reference

    from src.models.rosenblatt_perceptron.core import train_rosenblatt

    random = Random(57)
    examples = [
        ([random.gauss(0.0, 1.0) for _ in range(dimensions)], random.choice((-1, 1)))
        for _ in range(9)
    ]
    original = copy.deepcopy(examples)
    for budget in (1, 2, 7, 20):
        expected = reference(examples, budget, threshold, clamped)
        assert train_rosenblatt(examples, budget, threshold, clamped) == expected
        inputs, labels = zip(*examples)
        model, stats = train(
            inputs, labels, epochs=budget, threshold=threshold, clamped=clamped
        )
        actual = (
            model.weights + [model.bias] if clamped else model.weights,
            stats["epochs_run"],
            stats["updates"],
            stats["converged"],
            stats["strengths"],
        )
        assert actual == expected
        assert len(stats["history"]) == stats["epochs_run"]
        assert stats["updates"] == sum(stats["strengths"])
    assert examples == original


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_thresholds_remain_invalid(threshold):
    with pytest.raises(ValueError):
        train([[1.0]], [1], threshold=threshold)
