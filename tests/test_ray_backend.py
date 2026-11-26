import json
import os

import numpy as np
import pytest


def test_real_ray_worker_matches_local_mixed_activation_training(tmp_path):
    from src.models.neural_network import MultiLayerPerceptron
    from src.models.ray_backend import fit_with_ray

    x = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    y = np.array([0, 1, 2, 3])
    constructor = {
        "shape": [2, 6, 4, 4],
        "activation_functions": ["relu", "tanh", "softmax"],
        "seed": 11,
    }
    options = {"epochs": 12, "learning_rate": 0.05, "batch_size": 3, "tolerance": 0.0}
    local = MultiLayerPerceptron(**constructor).fit(x, y, **options)
    payload, execution = fit_with_ray(
        "multilayer_perceptron",
        x,
        y,
        constructor,
        options,
        num_cpus=1,
        cpus_per_task=1,
        timeout_seconds=60,
        temp_directory=None,
    )
    assert execution["backend"] == "ray"
    assert execution["worker_pid"] != os.getpid()
    assert execution["worker_id"]
    assert payload == local.to_dict()
    restored = MultiLayerPerceptron.from_dict(json.loads(json.dumps(payload)))
    np.testing.assert_array_equal(restored.predict_proba(x), local.predict_proba(x))
    import ray

    assert not ray.is_initialized()


@pytest.mark.skipif(
    os.name != "nt", reason="Native Windows Ray has MAX_PATH restrictions"
)
def test_long_windows_runtime_path_is_rejected_before_startup(tmp_path, monkeypatch):
    import ray

    from src.models.ray_backend import fit_with_ray

    def unexpected(*args, **kwargs):
        raise AssertionError("long paths must fail before Ray startup")

    monkeypatch.setattr(ray, "init", unexpected)
    root = tmp_path / ("nested_" * 12)
    with pytest.raises(ValueError, match="temp_directory.*long|shorter"):
        fit_with_ray(
            "logistic_regression",
            [[-1.0], [1.0]],
            [-1, 1],
            {"seed": 1},
            {"epochs": 1},
            temp_directory=root,
        )
    assert not root.exists()


def test_invalid_resources_fail_before_ray_startup(monkeypatch):
    import ray

    from src.models.ray_backend import fit_with_ray

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid resources must not start Ray")

    monkeypatch.setattr(ray, "init", unexpected)
    with pytest.raises(ValueError, match="cpus_per_task"):
        fit_with_ray(
            "logistic_regression",
            [[-1.0], [1.0]],
            [-1, 1],
            {"seed": 1},
            {"epochs": 1},
            num_cpus=1,
            cpus_per_task=2,
        )


def test_an_existing_ray_runtime_is_not_reused_or_shut_down(monkeypatch):
    import ray

    from src.models.ray_backend import fit_with_ray

    calls = []
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "shutdown", lambda: calls.append("shutdown"))
    monkeypatch.setattr(ray, "init", lambda **kw: calls.append("init"))
    with pytest.raises(RuntimeError, match="already initialized"):
        fit_with_ray(
            "logistic_regression", [[-1.0], [1.0]], [-1, 1], {"seed": 1}, {"epochs": 1}
        )
    assert calls == []


@pytest.mark.parametrize("failure", ["initialize", "task", "timeout"])
def test_failures_are_propagated_and_owned_runtime_is_cleaned(monkeypatch, failure):
    import ray

    from src.models.ray_backend import fit_with_ray

    calls = []
    reference = object()

    class Task:
        def remote(self, *args):
            calls.append("submitted")
            return reference

    def initialize(**kwargs):
        calls.append("initialized")
        if failure == "initialize":
            raise RuntimeError("synthetic initialization failure")

    def get(ref, timeout):
        assert ref is reference
        if failure == "timeout":
            raise ray.exceptions.GetTimeoutError("synthetic task timeout")
        raise ValueError("synthetic training failure")

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", initialize)
    monkeypatch.setattr(ray, "remote", lambda **kwargs: lambda function: Task())
    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(
        ray, "cancel", lambda ref, force: calls.append(("cancelled", force))
    )
    monkeypatch.setattr(ray, "shutdown", lambda: calls.append("shutdown"))
    with pytest.raises(RuntimeError, match="Ray training failed"):
        fit_with_ray(
            "logistic_regression", [[-1.0], [1.0]], [-1, 1], {"seed": 1}, {"epochs": 1}
        )
    assert calls[-1] == "shutdown"
    assert (("cancelled", True) in calls) is (failure == "timeout")
