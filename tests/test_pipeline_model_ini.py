"""Synthetic XML-to-model-profile boundary tests; never use the active pipeline."""

from pathlib import Path

import pytest

from src.pipeline.config import load_config
from src.pipeline.run import run_pipeline


def _pipeline(tmp_path, kind, options, *, enabled=False, preprocessing=""):
    path = tmp_path / "pipeline.xml"
    path.write_text(
        f'''<pipelineConfiguration>{preprocessing}<models>
    <model id="test" type="{kind}" run="{str(enabled).lower()}">
      <inputMatrix>missing.csv</inputMatrix><outputDir>output</outputDir>
      <targetColumn>aki_stage</targetColumn><features><column>preoperative/x</column></features>
      <split trainFraction="0.8" seed="2026"/><options>{options}</options>
    </model></models></pipelineConfiguration>''',
        encoding="utf8",
    )
    return path


@pytest.mark.parametrize(
    "kind", ["multilayer_perceptron", "multi_output_perceptron", "logistic_regression"]
)
def test_disabled_model_ini_paths_are_xml_relative_without_file_reads(
    tmp_path, kind, monkeypatch
):
    ini = tmp_path / "profiles" / "absent.ini"
    original = Path.open

    def guarded(self, *args, **kwargs):
        if self == ini:
            raise AssertionError("disabled model INI must not be opened")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    xml = _pipeline(
        tmp_path,
        kind,
        '<option name="config" value="profiles/absent.ini"/><option name="profile" value="experiment"/>',
    )
    step = load_config(xml).models[0]
    assert step.options["model_config"] == ini.resolve()
    assert step.options["profile"] == "experiment"
    assert run_pipeline(xml) == 0
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "options",
    [
        '<option name="profile" value="experiment"/>',
        '<option name="config" value="absent.ini"/><option name="epochs" value="1000"/>',
    ],
)
def test_xml_profiles_cannot_silently_mix_with_inline_training_options(
    tmp_path, options
):
    xml = _pipeline(tmp_path, "multilayer_perceptron", options)
    with pytest.raises(
        ValueError, match="requires config|cannot.*combin|authoritative"
    ):
        load_config(xml)


def test_enabled_ini_is_preflighted_before_any_preprocessing(tmp_path, monkeypatch):
    import src.pipeline.run as runner

    calls = []
    monkeypatch.setattr(
        runner, "_run_preprocessing", lambda config: calls.append("preprocessed")
    )
    (tmp_path / "models.ini").write_text("""[multilayer_perceptron]
model_type = multilayer_perceptron
hidden_layers = 8, 4
hidden_activations = relu
""")
    xml = _pipeline(
        tmp_path,
        "multilayer_perceptron",
        '<option name="config" value="models.ini"/>',
        enabled=True,
        preprocessing="<preprocessing/>",
    )
    with pytest.raises(ValueError, match="hidden_activations"):
        run_pipeline(xml)
    assert calls == []
    assert not (tmp_path / "output").exists()
