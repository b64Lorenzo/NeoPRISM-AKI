"""Synthetic-only contract tests for classifier INI settings."""

from dataclasses import FrozenInstanceError

import pytest

MODEL_TYPES = (
    "multilayer_perceptron",
    "multi_output_perceptron",
    "logistic_regression",
)


def write_ini(tmp_path, body):
    path = tmp_path / "models.ini"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", MODEL_TYPES)
def test_loads_immutable_model_specific_defaults(tmp_path, kind):
    from src.models.config import (
        ClassifierSettings,
        LogisticSettings,
        MLPSettings,
        MultiOutputSettings,
        OnlineSettings,
        RaySettings,
        TrainingSettings,
        load_classifier_settings,
    )

    path = write_ini(tmp_path, f"[ray]\n[{kind}]\nmodel_type = {kind}\n")
    settings = load_classifier_settings(path, model_type=kind)

    assert isinstance(settings, ClassifierSettings)
    assert settings.ray == RaySettings()
    assert settings.ray.num_cpus == 1
    assert settings.ray.cpus_per_task == 1.0
    assert settings.ray.timeout_seconds == 300.0
    assert settings.ray.temp_directory is None
    assert settings.model.model_type == kind
    assert settings.model.training.epochs == 1000
    assert settings.model.training.learning_rate == 0.05
    if kind == "multilayer_perceptron":
        assert isinstance(settings.model, MLPSettings)
        assert settings.model.training == TrainingSettings()
        assert settings.model.hidden_layers == (16,)
        assert settings.model.hidden_activations == ("relu",)
        assert settings.model.init_methods is None
        assert settings.model.output_activation == "softmax"
        assert settings.model.loss == "cross_entropy"
    elif kind == "multi_output_perceptron":
        assert isinstance(settings.model, MultiOutputSettings)
        assert settings.model.training == OnlineSettings()
        assert settings.model.output_activation == "linear"
        assert settings.model.loss == "mistake_rate"
        assert not hasattr(settings.model.training, "batch_size")
    else:
        assert isinstance(settings.model, LogisticSettings)
        assert settings.model.training == TrainingSettings()
        assert settings.model.l2 == 0.0
        assert settings.model.output_activation == "softmax"
        assert settings.model.loss == "cross_entropy"
    for target, attribute, value in (
        (settings.ray, "num_cpus", 2),
        (settings.model, "model_type", "other"),
        (settings.model.training, "epochs", 2),
        (settings, "model", None),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(target, attribute, value)


def test_loads_mixed_layer_activations_initializers_and_gradient_settings(tmp_path):
    from src.models.config import (
        MLPSettings,
        TrainingSettings,
        load_classifier_settings,
    )

    path = write_ini(
        tmp_path,
        """[ray]
[multilayer_perceptron]
model_type = multilayer_perceptron
hidden_layers = 12, 7
hidden_activations = leaky_relu, tanh
init_methods = he, xavier, zero
output_activation = softmax
loss = cross_entropy
epochs = 25
learning_rate = 2e-2
batch_size = 5
tolerance = .001
""",
    )
    model = load_classifier_settings(path, model_type="multilayer_perceptron").model
    assert isinstance(model, MLPSettings)
    assert model.hidden_layers == (12, 7)
    assert model.hidden_activations == ("leaky_relu", "tanh")
    assert model.init_methods == ("he", "xavier", "zero")
    assert model.training == TrainingSettings(25, 0.02, 5, 0.001)


@pytest.mark.parametrize(
    "extra, widths, activations, initializers",
    [
        ("hidden_layers = 8,4,2\n", (8, 4, 2), ("relu",) * 3, None),
        ("hidden_layers =\n", (), (), None),
        (
            "hidden_layers =\nhidden_activations =\ninit_methods = xavier\n",
            (),
            (),
            ("xavier",),
        ),
        ("hidden_layers = 2\nhidden_activations = sigmoid\n", (2,), ("sigmoid",), None),
        ("hidden_layers = 2\nhidden_activations = linear\n", (2,), ("linear",), None),
    ],
)
def test_mlp_architecture_defaults_depend_on_hidden_layer_count(
    tmp_path, extra, widths, activations, initializers
):
    from src.models.config import MLPSettings, load_classifier_settings

    path = write_ini(
        tmp_path,
        "[ray]\n[multilayer_perceptron]\nmodel_type = multilayer_perceptron\n" + extra,
    )
    model = load_classifier_settings(path, model_type="multilayer_perceptron").model
    assert isinstance(model, MLPSettings)
    assert model.hidden_layers == widths
    assert model.hidden_activations == activations
    assert model.init_methods == initializers


@pytest.mark.parametrize(
    "extra, field_name",
    [
        ("hidden_layers = 4,0", "hidden_layers"),
        ("hidden_layers = -1", "hidden_layers"),
        ("hidden_layers = 2,", "hidden_layers"),
        ("hidden_layers = 2,,3", "hidden_layers"),
        ("hidden_layers = 2.0", "hidden_layers"),
        ("hidden_layers = 2_0", "hidden_layers"),
        ("hidden_layers = 8,4\nhidden_activations = relu", "hidden_activations"),
        ("hidden_activations =", "hidden_activations"),
        ("hidden_activations = relu,", "hidden_activations"),
        ("hidden_activations = softmax", "hidden_activations"),
        ("hidden_activations = ReLU", "hidden_activations"),
        ("hidden_layers =\nhidden_activations = relu", "hidden_activations"),
        ("init_methods = he", "init_methods"),
        ("init_methods =", "init_methods"),
        ("init_methods = he,unknown", "init_methods"),
        ("init_methods = he,xavier,", "init_methods"),
        ("output_activation = sigmoid", "output_activation"),
        ("loss = binary_cross_entropy", "loss"),
        ("output_activation = linear\nloss = mse", "output_activation"),
    ],
)
def test_rejects_invalid_mlp_architecture(tmp_path, extra, field_name):
    from src.models.config import load_classifier_settings

    path = write_ini(
        tmp_path,
        "[ray]\n[multilayer_perceptron]\nmodel_type = multilayer_perceptron\n" + extra,
    )
    with pytest.raises(ValueError, match=field_name):
        load_classifier_settings(path, model_type="multilayer_perceptron")


def test_logistic_and_online_training_use_distinct_fields(tmp_path):
    from src.models.config import (
        LogisticSettings,
        OnlineSettings,
        TrainingSettings,
        load_classifier_settings,
    )

    path = write_ini(
        tmp_path,
        """[ray]
[logistic_regression]
model_type = logistic_regression
epochs = 12
learning_rate = .1
batch_size = 3
tolerance = 0.01
l2 = .25
output_activation = softmax
loss = cross_entropy
[multi_output_perceptron]
model_type = multi_output_perceptron
epochs = 7
learning_rate = .2
output_activation = linear
loss = mistake_rate
""",
    )
    logistic = load_classifier_settings(path, model_type="logistic_regression").model
    online = load_classifier_settings(path, model_type="multi_output_perceptron").model
    assert isinstance(logistic, LogisticSettings)
    assert logistic.training == TrainingSettings(12, 0.1, 3, 0.01)
    assert logistic.l2 == 0.25
    assert online.training == OnlineSettings(7, 0.2)


@pytest.mark.parametrize(
    "kind, extra, field_name",
    [
        ("logistic_regression", "l2 = -1", "l2"),
        ("logistic_regression", "output_activation = sigmoid", "output_activation"),
        ("logistic_regression", "loss = mse", "loss"),
        ("multi_output_perceptron", "output_activation = softmax", "output_activation"),
        ("multi_output_perceptron", "loss = cross_entropy", "loss"),
    ],
)
def test_rejects_incompatible_model_objectives(tmp_path, kind, extra, field_name):
    from src.models.config import load_classifier_settings

    path = write_ini(tmp_path, f"[ray]\n[{kind}]\nmodel_type = {kind}\n{extra}\n")
    with pytest.raises(ValueError, match=field_name):
        load_classifier_settings(path, model_type=kind)


def test_ray_resources_and_relative_temporary_directory_are_ini_relative(
    tmp_path, monkeypatch
):
    from src.models.config import RaySettings, load_classifier_settings

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = write_ini(
        tmp_path,
        """[ray]
num_cpus = 3
cpus_per_task = .5
timeout_seconds = 12.5
temp_directory = local-ray/not-created
[logistic_regression]
model_type = logistic_regression
""",
    )
    monkeypatch.chdir(elsewhere)
    settings = load_classifier_settings(path, model_type="logistic_regression")
    assert settings.ray == RaySettings(3, 0.5, 12.5, tmp_path / "local-ray/not-created")
    assert settings.ray.temp_directory is not None
    assert not settings.ray.temp_directory.exists()
    assert set(tmp_path.iterdir()) == {path, elsewhere}


@pytest.mark.parametrize("temporary", ["", "   "])
def test_empty_ray_temporary_directory_means_none(tmp_path, temporary):
    from src.models.config import load_classifier_settings

    path = write_ini(
        tmp_path,
        f"[ray]\ntemp_directory = {temporary}\n[logistic_regression]\n"
        "model_type = logistic_regression\n",
    )
    assert (
        load_classifier_settings(
            path, model_type="logistic_regression"
        ).ray.temp_directory
        is None
    )


def test_absolute_ray_temporary_directory_is_not_rebased(tmp_path):
    from src.models.config import load_classifier_settings

    temporary = (tmp_path / "unused").resolve()
    path = write_ini(
        tmp_path,
        f"[ray]\ntemp_directory = {temporary}\n[logistic_regression]\n"
        "model_type = logistic_regression\n",
    )
    assert (
        load_classifier_settings(
            path, model_type="logistic_regression"
        ).ray.temp_directory
        == temporary
    )
    assert not temporary.exists()


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("num_cpus", "0"),
        ("num_cpus", "-1"),
        ("num_cpus", "1.5"),
        ("num_cpus", "1_0"),
        ("cpus_per_task", "0"),
        ("cpus_per_task", "-1"),
        ("cpus_per_task", "2"),
        ("cpus_per_task", "NaN"),
        ("timeout_seconds", "0"),
        ("timeout_seconds", "inf"),
        ("timeout_seconds", "1e999"),
        ("timeout_seconds", "2 seconds"),
    ],
)
def test_rejects_invalid_ray_resources(tmp_path, field_name, value):
    from src.models.config import load_classifier_settings

    path = write_ini(
        tmp_path,
        f"[ray]\n{field_name} = {value}\n[logistic_regression]\n"
        "model_type = logistic_regression\n",
    )
    with pytest.raises(ValueError, match=field_name):
        load_classifier_settings(path, model_type="logistic_regression")


def test_named_profile_selects_declared_model_type(tmp_path):
    from src.models.config import MLPSettings, load_classifier_settings

    path = write_ini(
        tmp_path,
        """[small]
model_type = multilayer_perceptron
hidden_layers = 3,2
[logistic_regression]
model_type = logistic_regression
""",
    )
    settings = load_classifier_settings(
        path, model_type="multilayer_perceptron", profile="small"
    )
    assert isinstance(settings.model, MLPSettings)
    assert settings.model.hidden_layers == (3, 2)
    assert settings.ray.num_cpus == 1


@pytest.mark.parametrize(
    "body, kind, profile, diagnostic",
    [
        ("[logistic_regression]\n", "logistic_regression", None, "model_type"),
        (
            "[logistic_regression]\nmodel_type =\n",
            "logistic_regression",
            None,
            "model_type",
        ),
        (
            "[other]\nmodel_type = unknown\n",
            "logistic_regression",
            "other",
            "model_type",
        ),
        (
            "[other]\nmodel_type = multilayer_perceptron\n",
            "logistic_regression",
            "other",
            "match",
        ),
        (
            "[logistic_regression]\nmodel_type = logistic_regression\n",
            "unknown",
            None,
            "model_type",
        ),
        (
            "[logistic_regression]\nmodel_type = logistic_regression\n",
            "logistic_regression",
            "missing",
            "profile",
        ),
        (
            "[logistic_regression]\nmodel_type = logistic_regression\n",
            "logistic_regression",
            "",
            "profile",
        ),
        ("[ray]\n", "logistic_regression", "ray", "profile"),
        ("", "logistic_regression", None, "profile"),
    ],
)
def test_rejects_missing_or_mismatched_profiles(
    tmp_path, body, kind, profile, diagnostic
):
    from src.models.config import load_classifier_settings

    path = write_ini(tmp_path, body)
    with pytest.raises(ValueError, match=diagnostic):
        load_classifier_settings(path, model_type=kind, profile=profile)


@pytest.mark.parametrize(
    "other",
    [
        "[broken]\nmodel_type = multilayer_perceptron\nhidden_layers = 0\n",
        "[broken]\nmodel_type = logistic_regression\nepochs = 0\n",
        "[broken]\nmodel_type = multi_output_perceptron\nbatch_size = 1\n",
        "[broken]\nmodel_type = unknown\n",
        "[broken]\nepochs = 10\n",
    ],
)
def test_validates_unselected_profiles_before_returning(tmp_path, other):
    from src.models.config import load_classifier_settings

    path = write_ini(
        tmp_path,
        "[logistic_regression]\nmodel_type = logistic_regression\n" + other,
    )
    with pytest.raises(ValueError):
        load_classifier_settings(path, model_type="logistic_regression")


@pytest.mark.parametrize(
    "kind, key",
    [
        ("ray", "address"),
        ("ray", "backend"),
        ("ray", "num_gpus"),
        ("ray", "unknown"),
        ("logistic_regression", "hidden_layers"),
        ("logistic_regression", "hidden_activations"),
        ("logistic_regression", "init_methods"),
        ("multilayer_perceptron", "l2"),
        ("multilayer_perceptron", "activation"),
        ("multi_output_perceptron", "hidden_layers"),
        ("multi_output_perceptron", "batch_size"),
        ("multi_output_perceptron", "tolerance"),
        ("multi_output_perceptron", "l2"),
        ("logistic_regression", "input"),
        ("logistic_regression", "output_dir"),
        ("logistic_regression", "features"),
        ("logistic_regression", "seed"),
        ("logistic_regression", "train_fraction"),
        ("logistic_regression", "Epochs"),
    ],
)
def test_rejects_unknown_or_wrong_model_keys(tmp_path, kind, key):
    from src.models.config import load_classifier_settings

    body = (
        f"[ray]\n{key} = 0\n[logistic_regression]\nmodel_type = logistic_regression\n"
        if kind == "ray"
        else f"[{kind}]\nmodel_type = {kind}\n{key} = 0\n"
    )
    path = write_ini(tmp_path, body)
    with pytest.raises(ValueError, match=key):
        load_classifier_settings(
            path, model_type="logistic_regression" if kind == "ray" else kind
        )


@pytest.mark.parametrize(
    "prefix",
    [
        "[DEFAULT]\n",
        "[DEFAULT]\nepochs = 2\n",
    ],
)
def test_default_section_is_forbidden_even_when_empty(tmp_path, prefix):
    from src.models.config import load_classifier_settings

    path = write_ini(
        tmp_path, prefix + "[logistic_regression]\nmodel_type = logistic_regression\n"
    )
    with pytest.raises(ValueError, match="DEFAULT"):
        load_classifier_settings(path, model_type="logistic_regression")


@pytest.mark.parametrize(
    "body",
    [
        "[logistic_regression]\nmodel_type = logistic_regression\nepochs = 1\nepochs = 2\n",
        "[logistic_regression]\nmodel_type = logistic_regression\n[logistic_regression]\n",
        "[logistic_regression]\nmodel_type = logistic_regression\nepochs\n",
        "not an INI\n",
    ],
)
def test_malformed_ini_is_reported_as_value_error(tmp_path, body):
    from src.models.config import load_classifier_settings

    with pytest.raises(ValueError, match="INI"):
        load_classifier_settings(
            write_ini(tmp_path, body), model_type="logistic_regression"
        )


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("epochs", ""),
        ("epochs", "0"),
        ("epochs", "2_0"),
        ("epochs", "2.0"),
        ("epochs", "2e2"),
        ("epochs", "2x"),
        ("epochs", "true"),
        ("epochs", "２"),
        ("batch_size", "-1"),
        ("learning_rate", "0"),
        ("learning_rate", "-0.5"),
        ("learning_rate", "nan"),
        ("learning_rate", "Infinity"),
        ("learning_rate", "1e999"),
        ("learning_rate", "1e-999"),
        ("learning_rate", "0.05 # inline comment"),
        ("learning_rate", "%(rate)s"),
        ("tolerance", "-1"),
        ("tolerance", "inf"),
        ("l2", "nan"),
    ],
)
def test_numeric_tokens_are_complete_finite_and_in_range(tmp_path, field_name, value):
    from src.models.config import load_classifier_settings

    path = write_ini(
        tmp_path,
        "[logistic_regression]\nmodel_type = logistic_regression\n"
        f"{field_name} = {value}\n",
    )
    with pytest.raises(ValueError, match=field_name):
        load_classifier_settings(path, model_type="logistic_regression")


def test_utf8_bom_is_supported(tmp_path):
    from src.models.config import load_classifier_settings

    path = tmp_path / "models.ini"
    path.write_text(
        "[logistic_regression]\nmodel_type = logistic_regression\n",
        encoding="utf-8-sig",
    )
    assert (
        load_classifier_settings(
            path, model_type="logistic_regression"
        ).model.model_type
        == "logistic_regression"
    )


@pytest.mark.parametrize("temporary", ["~/ray", "$HOME/ray", "%(ray_root)s"])
def test_temporary_directory_has_no_interpolation_or_environment_expansion(
    tmp_path, temporary
):
    from src.models.config import load_classifier_settings

    path = write_ini(
        tmp_path,
        f"[ray]\ntemp_directory = {temporary}\n[logistic_regression]\n"
        "model_type = logistic_regression\n",
    )
    settings = load_classifier_settings(path, model_type="logistic_regression")
    assert settings.ray.temp_directory == (tmp_path / temporary).resolve()
    assert set(tmp_path.iterdir()) == {path}


def test_missing_file_is_not_silently_ignored(tmp_path):
    from src.models.config import load_classifier_settings

    with pytest.raises(FileNotFoundError):
        load_classifier_settings(
            tmp_path / "missing.ini", model_type="logistic_regression"
        )
    assert list(tmp_path.iterdir()) == []


def test_parser_loads_without_site_packages_or_training_imports(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    path = write_ini(
        tmp_path, "[logistic_regression]\nmodel_type = logistic_regression\n"
    )
    probe = """import sys
from pathlib import Path
from src.models.config import load_classifier_settings
settings = load_classifier_settings(sys.argv[1], model_type="logistic_regression")
assert settings.model.model_type == "logistic_regression"
assert "ray" not in sys.modules
assert "numpy" not in sys.modules
assert "src.models.classification" not in sys.modules
assert list(Path(sys.argv[1]).parent.iterdir()) == [Path(sys.argv[1])]
print("stdlib-only settings load passed")
"""
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-c", probe, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "stdlib-only settings load passed"


@pytest.mark.parametrize("kind", MODEL_TYPES)
def test_commented_example_loads_every_canonical_profile(kind):
    from pathlib import Path

    from src.models.config import load_classifier_settings

    path = Path(__file__).resolve().parents[1] / "config/models.example.ini"
    settings = load_classifier_settings(path, model_type=kind)
    assert settings.model.model_type == kind
    assert settings.ray.num_cpus == 1
    assert settings.ray.cpus_per_task == 1.0
    assert settings.ray.timeout_seconds == 300.0
    assert settings.ray.temp_directory is None
