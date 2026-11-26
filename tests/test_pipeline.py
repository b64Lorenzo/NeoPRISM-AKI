"""Synthetic-only checks for XML orchestration and extension boundaries."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def write_xml(tmp_path, body, *, name="pipeline.xml", encoding="utf-8"):
    path = tmp_path / name
    path.write_text(body, encoding=encoding)
    return path


def test_empty_pipeline_is_valid(tmp_path):
    from src.pipeline.config import load_config

    config = load_config(write_xml(tmp_path, "<pipelineConfiguration/>"))

    assert config.preprocessing is None
    assert config.models == ()


def test_preprocessing_paths_are_xml_relative_and_disabled_by_default(tmp_path):
    from src.pipeline.config import load_config

    path = write_xml(
        tmp_path,
        """<pipelineConfiguration><preprocessing>
      <extraction config="parameters.ini" outputDir="outputs">
        <input>input.csv</input><input>other</input>
      </extraction>
      <plotting config="plotting.ini"/>
      <inputsModel config="inputs.ini" unlabeled="true"/>
    </preprocessing></pipelineConfiguration>""",
    )
    config = load_config(path)
    stages = config.preprocessing

    assert config.path == path.resolve()
    assert stages.extraction.inputs == (tmp_path / "input.csv", tmp_path / "other")
    assert stages.extraction.config == tmp_path / "parameters.ini"
    assert stages.extraction.output_dir == tmp_path / "outputs"
    assert stages.plotting.config == tmp_path / "plotting.ini"
    assert stages.inputs_model.config == tmp_path / "inputs.ini"
    assert stages.inputs_model.unlabeled is True
    assert not stages.extraction.run
    assert not stages.plotting.run
    assert not stages.inputs_model.run


@pytest.mark.parametrize(
    "body",
    [
        "<wrong/>",
        "<pipelineConfiguration><general/></pipelineConfiguration>",
        "<pipelineConfiguration><preprocessing/><preprocessing/></pipelineConfiguration>",
        '<pipelineConfiguration><preprocessing><plotting config="x" run="yes"/></preprocessing></pipelineConfiguration>',
        '<pipelineConfiguration><preprocessing><plotting config=""/></preprocessing></pipelineConfiguration>',
        '<pipelineConfiguration><preprocessing><extraction config="x" outputDir="y"/></preprocessing></pipelineConfiguration>',
        '<pipelineConfiguration><preprocessing><plotting config="x"/><extraction config="x" outputDir="y"><input>x</input></extraction></preprocessing></pipelineConfiguration>',
        '<pipelineConfiguration xmlns="urn:unexpected"/>',
        "<pipelineConfiguration>unexpected</pipelineConfiguration>",
        "<pipelineConfiguration><preprocessing/>unexpected</pipelineConfiguration>",
        "<pipelineConfiguration",
    ],
)
def test_invalid_structure_fails_as_value_error(tmp_path, body):
    from src.pipeline.config import load_config

    with pytest.raises(ValueError):
        load_config(write_xml(tmp_path, body))


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_dtd_is_rejected_before_entity_expansion(tmp_path, encoding):
    from src.pipeline.config import load_config

    path = write_xml(
        tmp_path,
        """<!DOCTYPE pipelineConfiguration [
      <!ENTITY test "expanded">
    ]><pipelineConfiguration>&test;</pipelineConfiguration>""",
        encoding=encoding,
    )
    with pytest.raises(ValueError, match="DTD"):
        load_config(path)


def test_byte_limit_includes_exact_boundary(tmp_path):
    from src.pipeline.config import load_config

    path = write_xml(tmp_path, "<pipelineConfiguration/>")
    size = path.stat().st_size
    assert load_config(path, max_bytes=size).models == ()
    with pytest.raises(ValueError, match="size limit"):
        load_config(path, max_bytes=size - 1)


def model_xml(
    *,
    id="one",
    type="logistic_regression",
    run="false",
    output="outputs/one",
    input="matrix.csv",
    target="aki_label",
    features="<column>age</column>",
    options="",
):
    return f'''<model id="{id}" type="{type}" run="{run}">
      <inputMatrix>{input}</inputMatrix><outputDir>{output}</outputDir>
      <targetColumn>{target}</targetColumn><features>{features}</features>
      <split trainFraction="0.8" seed="2026"/><options>{options}</options>
    </model>'''


def models_xml(*models):
    return (
        "<pipelineConfiguration><models>"
        + "".join(models)
        + "</models></pipelineConfiguration>"
    )


def test_models_own_their_paths_features_and_split(tmp_path):
    from src.pipeline.config import load_config

    config = load_config(
        write_xml(
            tmp_path,
            models_xml(
                model_xml(),
                model_xml(
                    id="second",
                    output="outputs/two",
                    input="another.csv",
                    target="aki_stage",
                    features="<column>weight</column><column>sex</column>",
                ),
            ),
        )
    )
    first, second = (step.config for step in config.models)
    assert first.model_type == second.model_type == "logistic_regression"
    assert first.id == "one" and second.id == "second"
    assert first.input_matrix == tmp_path / "matrix.csv"
    assert second.input_matrix == tmp_path / "another.csv"
    assert first.output_dir == tmp_path / "outputs/one"
    assert second.output_dir == tmp_path / "outputs/two"
    assert first.features == ("age",)
    assert second.features == ("weight", "sex")
    assert second.target_column == "aki_stage"
    assert first.train_fraction == 0.8 and first.seed == 2026
    assert not first.run and not second.run
    assert not first.input_matrix.exists()


@pytest.mark.parametrize(
    "type, expected",
    [
        (
            "rosenblatt_perceptron",
            {"epochs": 100, "threshold": 0.0, "clamped": False},
        ),
        (
            "logistic_regression",
            {
                "epochs": 1000,
                "learning_rate": 0.05,
                "batch_size": 32,
                "tolerance": 0.0,
                "l2": 0.0,
            },
        ),
        (
            "multilayer_perceptron",
            {
                "epochs": 1000,
                "learning_rate": 0.05,
                "batch_size": 32,
                "tolerance": 0.0,
                "hidden_layers": (16,),
                "activation": "relu",
            },
        ),
    ],
)
def test_builtin_options_have_model_specific_typed_defaults(tmp_path, type, expected):
    from src.pipeline.config import load_config
    from src.pipeline.registry import ModelStep

    step = load_config(write_xml(tmp_path, models_xml(model_xml(type=type)))).models[0]
    assert isinstance(step, ModelStep)
    assert dict(step.options) == expected
    with pytest.raises(TypeError):
        step.options["epochs"] = 1


def test_custom_registry_pairs_typed_options_and_handler(tmp_path):
    from dataclasses import dataclass

    from src.pipeline.config import load_config
    from src.pipeline.registry import ModelRegistry

    @dataclass(frozen=True)
    class Options:
        factor: int

    calls = []

    def parse(config, raw):
        assert config.model_type == "custom_scale"
        return Options(int(raw["factor"]))

    def execute(config, options):
        calls.append((config.id, options.factor))
        return 0

    registry = ModelRegistry()
    registry.register("custom_scale", parse, execute)
    path = write_xml(
        tmp_path,
        models_xml(
            model_xml(
                type="custom_scale",
                options='<option name="factor" value="3"/>',
            )
        ),
    )
    step = load_config(path, registry).models[0]
    assert calls == []
    assert step.execute() == 0
    assert calls == [("one", 3)]
    with pytest.raises(ValueError, match="already registered"):
        registry.register("custom_scale", parse, execute)
    with pytest.raises(TypeError, match="callable"):
        registry.register("missing_handler", parse, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fragment",
    [
        model_xml(type="unknown"),
        model_xml(target="unsupported"),
        model_xml(type="rosenblatt_perceptron", target="aki_stage"),
        model_xml(features=""),
        model_xml(features="<column>age</column><column>age</column>"),
        model_xml(features="<column>aki_label</column>"),
        model_xml().replace("<options></options>", ""),
        model_xml().replace("<inputMatrix>matrix.csv</inputMatrix>", ""),
        model_xml().replace('trainFraction="0.8"', 'trainFraction="1"'),
        model_xml().replace('trainFraction="0.8"', 'trainFraction="NaN"'),
        model_xml().replace('seed="2026"', 'seed="-1"'),
        model_xml().replace('seed="2026"', 'seed="2_026"'),
        model_xml().replace("<features>", '<features wrong="true">'),
        model_xml().replace(
            "<column>age</column>", "<column><name>age</name></column>"
        ),
        model_xml().replace(
            "<outputDir>outputs/one</outputDir>", "<outputDir> </outputDir>"
        ),
    ],
)
def test_invalid_model_configuration_is_rejected_even_when_disabled(tmp_path, fragment):
    from src.pipeline.config import load_config

    with pytest.raises(ValueError):
        load_config(write_xml(tmp_path, models_xml(fragment)))


@pytest.mark.parametrize(
    "type, name, value",
    [
        ("logistic_regression", "epochs", "0"),
        ("logistic_regression", "epochs", "1.5"),
        ("logistic_regression", "learning_rate", "NaN"),
        ("logistic_regression", "learning_rate", "inf"),
        ("logistic_regression", "learning_rate", "0"),
        ("logistic_regression", "batch_size", "0"),
        ("logistic_regression", "tolerance", "-1"),
        ("logistic_regression", "l2", "-1"),
        ("logistic_regression", "threshold", "0"),
        ("rosenblatt_perceptron", "threshold", "inf"),
        ("rosenblatt_perceptron", "batch_size", "10"),
        ("rosenblatt_perceptron", "learning_rate", "-0.1"),
        ("multilayer_perceptron", "l2", "1"),
        ("multilayer_perceptron", "hidden_layers", "16,0"),
        ("multilayer_perceptron", "hidden_layers", "16,"),
        ("multilayer_perceptron", "activation", "unknown"),
        ("multilayer_perceptron", "unknown", "1"),
    ],
)
def test_invalid_model_options_are_rejected(tmp_path, type, name, value):
    from src.pipeline.config import load_config

    fragment = model_xml(type=type, options=f'<option name="{name}" value="{value}"/>')
    with pytest.raises(
        ValueError,
        match="option|epochs|learning_rate|batch_size|tolerance|l2|threshold|hidden_layers|activation",
    ):
        load_config(write_xml(tmp_path, models_xml(fragment)))


def test_duplicate_option_names_are_rejected(tmp_path):
    from src.pipeline.config import load_config

    fragment = model_xml(
        options='<option name="epochs" value="2"/><option name="epochs" value="3"/>'
    )
    with pytest.raises(ValueError, match="Duplicate option"):
        load_config(write_xml(tmp_path, models_xml(fragment)))


def test_duplicate_ids_are_rejected(tmp_path):
    from src.pipeline.config import load_config

    with pytest.raises(ValueError, match="Duplicate model id"):
        load_config(
            write_xml(tmp_path, models_xml(model_xml(), model_xml(output="different")))
        )


def test_enabled_output_duplicates_are_resolved_before_comparison(tmp_path):
    from src.pipeline.config import load_config

    second = model_xml(id="two", run="true", output="other/../outputs/one")
    with pytest.raises(ValueError, match="output directory"):
        load_config(write_xml(tmp_path, models_xml(model_xml(run="true"), second)))
    second = model_xml(id="two", run="false", output="outputs/one")
    assert (
        len(
            load_config(
                write_xml(tmp_path, models_xml(model_xml(run="true"), second))
            ).models
        )
        == 2
    )


def recording_registry(calls, *, result=0):
    from src.pipeline.registry import ModelRegistry

    registry = ModelRegistry()

    def parse(config, raw):
        if set(raw) - {"multiplier"}:
            raise ValueError("Unknown synthetic option")
        return int(raw.get("multiplier", "1"))

    def execute(config, multiplier):
        import csv
        import json

        with config.input_matrix.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        calls.append(config.id)
        config.output_dir.mkdir(parents=True)
        (config.output_dir / "synthetic.json").write_text(
            json.dumps({"id": config.id, "rows": len(rows), "multiplier": multiplier}),
            encoding="utf-8",
        )
        return result

    registry.register("synthetic", parse, execute)
    return registry


def test_run_pipeline_executes_repeated_custom_types_in_xml_order(tmp_path):
    import json

    from src.pipeline.run import run_pipeline

    calls = []
    (tmp_path / "first.csv").write_text("age,aki_label\n1,-1\n2,1\n", encoding="utf-8")
    (tmp_path / "second.csv").write_text("age,aki_label\n3,1\n", encoding="utf-8")
    path = write_xml(
        tmp_path,
        models_xml(
            model_xml(
                id="first",
                type="synthetic",
                run="true",
                input="first.csv",
                output="first_out",
                options='<option name="multiplier" value="3"/>',
            ),
            model_xml(
                id="disabled",
                type="synthetic",
                input="absent.csv",
                output="never_created",
            ),
            model_xml(
                id="second",
                type="synthetic",
                run="true",
                input="second.csv",
                output="second_out",
            ),
        ),
    )
    assert run_pipeline(path, recording_registry(calls)) == 0
    assert calls == ["first", "second"]
    assert json.loads((tmp_path / "first_out/synthetic.json").read_text()) == {
        "id": "first",
        "rows": 2,
        "multiplier": 3,
    }
    assert json.loads((tmp_path / "second_out/synthetic.json").read_text())["rows"] == 1
    assert not (tmp_path / "never_created").exists()


def install_stage_handlers(monkeypatch, handler):
    import sys
    from types import SimpleNamespace

    for name in ("parameters", "plotting", "inputsModel"):
        monkeypatch.setitem(
            sys.modules,
            f"src.preprocessing.{name}.run",
            SimpleNamespace(
                run=lambda args, name=name: handler(name, args),
            ),
        )


def preprocessing_xml(*, extraction="true", plotting="true", inputs="true"):
    return f'''<preprocessing>
      <extraction run="{extraction}" config="parameters.ini" outputDir="extracted">
        <input>raw.csv</input>
      </extraction>
      <plotting run="{plotting}" config="plotting.ini"/>
      <inputsModel run="{inputs}" config="inputs.ini" unlabeled="true"/>
    </preprocessing>'''


def test_preprocessing_runs_in_order_and_can_create_later_matrix(tmp_path, monkeypatch):
    from src.pipeline.run import run_pipeline

    calls = []
    matrix = tmp_path / "matrix.csv"

    def stage(name, args):
        calls.append(name)
        if name == "parameters":
            assert args.input == [tmp_path / "raw.csv"]
            assert args.output_dir == tmp_path / "extracted"
            assert args.config == tmp_path / "parameters.ini"
        elif name == "plotting":
            assert vars(args) == {"config": tmp_path / "plotting.ini"}
        else:
            assert vars(args) == {"config": tmp_path / "inputs.ini", "unlabeled": True}
            matrix.write_text("age,aki_label\n1,1\n", encoding="utf-8")
        return 0

    install_stage_handlers(monkeypatch, stage)
    path = write_xml(
        tmp_path,
        "<pipelineConfiguration>"
        + preprocessing_xml()
        + "<models>"
        + model_xml(type="synthetic", run="true")
        + "</models></pipelineConfiguration>",
    )
    assert not matrix.exists()
    assert run_pipeline(path, recording_registry(calls)) == 0
    assert calls == ["parameters", "plotting", "inputsModel", "one"]


def test_disabled_preprocessing_stages_are_skipped(tmp_path, monkeypatch):
    from src.pipeline.run import run_pipeline

    calls = []
    install_stage_handlers(monkeypatch, lambda name, args: calls.append(name))
    path = write_xml(
        tmp_path,
        "<pipelineConfiguration>"
        + preprocessing_xml(extraction="false", inputs="false")
        + "</pipelineConfiguration>",
    )
    assert run_pipeline(path) == 0
    assert calls == ["plotting"]


def test_later_model_validation_happens_before_any_execution(tmp_path, monkeypatch):
    from src.pipeline.run import run_pipeline

    calls = []
    install_stage_handlers(monkeypatch, lambda name, args: calls.append(name))
    path = write_xml(
        tmp_path,
        "<pipelineConfiguration>"
        + preprocessing_xml()
        + "<models>"
        + model_xml(id="valid", type="synthetic", run="true")
        + model_xml(
            id="bad",
            type="synthetic",
            run="true",
            output="other",
            options='<option name="unknown" value="1"/>',
        )
        + "</models></pipelineConfiguration>",
    )
    with pytest.raises(ValueError, match="Unknown synthetic option"):
        run_pipeline(path, recording_registry(calls))
    assert calls == []
    assert not (tmp_path / "outputs").exists()


def test_stage_nonzero_exit_stops_execution(tmp_path, monkeypatch):
    from src.pipeline.run import run_pipeline

    calls = []

    def stage(name, args):
        calls.append(name)
        return 7

    install_stage_handlers(monkeypatch, stage)
    path = write_xml(
        tmp_path,
        "<pipelineConfiguration>" + preprocessing_xml() + "</pipelineConfiguration>",
    )
    assert run_pipeline(path) == 7
    assert calls == ["parameters"]


def test_model_nonzero_exit_stops_execution(tmp_path):
    from src.pipeline.run import run_pipeline

    calls = []
    (tmp_path / "matrix.csv").write_text("age,aki_label\n1,1\n", encoding="utf-8")
    path = write_xml(
        tmp_path,
        models_xml(
            model_xml(type="synthetic", run="true"),
            model_xml(id="two", type="synthetic", run="true", output="other"),
        ),
    )
    assert run_pipeline(path, recording_registry(calls, result=5)) == 5
    assert calls == ["one"]
    assert not (tmp_path / "other").exists()


def test_missing_matrix_is_checked_immediately_before_model_execution(tmp_path):
    from src.pipeline.run import run_pipeline

    calls = []
    path = write_xml(tmp_path, models_xml(model_xml(type="synthetic", run="true")))
    with pytest.raises(ValueError, match="matrix"):
        run_pipeline(path, recording_registry(calls))
    assert calls == []
    assert not (tmp_path / "outputs").exists()


def test_disabled_pipeline_has_no_domain_imports_or_output_side_effects(tmp_path):
    import subprocess
    import sys

    path = write_xml(
        tmp_path,
        "<pipelineConfiguration>"
        + preprocessing_xml(extraction="false", plotting="false", inputs="false")
        + "<models>"
        + model_xml()
        + "</models></pipelineConfiguration>",
    )
    probe = """import sys
from src.pipeline.run import run_pipeline
assert run_pipeline(sys.argv[1]) == 0
assert not any(name.startswith("src.preprocessing.") for name in sys.modules)
assert "src.models.classification" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {item.name for item in tmp_path.iterdir()} == {"pipeline.xml"}


def test_register_parser_exposes_pipeline_command(tmp_path):
    import argparse

    from src.pipeline.run import register_parser

    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(dest="command", required=True))
    path = write_xml(tmp_path, "<pipelineConfiguration/>")
    args = parser.parse_args(["pipeline", "--config", str(path)])
    assert args.handler(args) == 0
    assert parser.parse_args(["pipeline"]).config == Path("config/pipeline.xml")


def test_default_model_handler_forwards_exact_fit_contract(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    from src.pipeline.run import run_pipeline

    calls = []
    module = ModuleType("src.models.classification")

    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        return {"synthetic_contract_probe": True}

    module.__dict__["fit"] = fit
    monkeypatch.setitem(sys.modules, module.__name__, module)
    (tmp_path / "matrix.csv").write_text("age,aki_label\n1,1\n", encoding="utf-8")
    path = write_xml(tmp_path, models_xml(model_xml(run="true")))
    assert run_pipeline(path) == 0
    assert calls == [
        (
            (tmp_path / "matrix.csv", tmp_path / "outputs/one"),
            {
                "model_type": "logistic_regression",
                "target_column": "aki_label",
                "features": ("age",),
                "train_fraction": 0.8,
                "seed": 2026,
                "epochs": 1000,
                "learning_rate": 0.05,
                "batch_size": 32,
                "tolerance": 0.0,
                "l2": 0.0,
            },
        )
    ]


@pytest.mark.parametrize("name", ["pipeline.xml", "pipeline.example.xml"])
def test_checked_in_templates_are_disabled_and_load_without_domain_files(name):
    from src.pipeline.config import load_config
    from src.pipeline.run import run_pipeline

    path = ROOT / "config" / name
    config = load_config(path)
    stages = config.preprocessing
    assert stages is not None
    assert stages.extraction is not None and not stages.extraction.run
    assert stages.plotting is not None and not stages.plotting.run
    assert stages.inputs_model is not None and not stages.inputs_model.run
    assert [step.config.model_type for step in config.models] == [
        "rosenblatt_perceptron",
        "logistic_regression",
        "multilayer_perceptron",
    ]
    assert all(not step.config.run for step in config.models)
    assert run_pipeline(path) == 0


def test_xsd_validates_both_templates_and_extensible_type_contract():
    etree = pytest.importorskip(
        "lxml.etree", reason="XSD validation is optional, not a runtime dependency"
    )
    schema = etree.XMLSchema(etree.parse(str(ROOT / "config/pipeline.xsd")))
    for name in ("pipeline.xml", "pipeline.example.xml"):
        schema.assertValid(etree.parse(str(ROOT / "config" / name)))
    schema.assertValid(
        etree.fromstring(models_xml(model_xml(type="future_model")).encode())
    )
    for document in (
        "<wrong/>",
        models_xml(model_xml(), model_xml(output="other")),
        "<pipelineConfiguration><models/><preprocessing/></pipelineConfiguration>",
    ):
        assert not schema.validate(etree.fromstring(document.encode()))
