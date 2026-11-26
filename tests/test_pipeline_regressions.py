import pytest

from src.cli import build_parser
from src.pipeline.config import load_config


def _xml(
    tmp_path, *, kind="multilayer_perceptron", feature="preoperative/age", options=""
):
    path = tmp_path / "pipeline.xml"
    path.write_text(
        f'''<pipelineConfiguration><models>
<model id="baseline" type="{kind}" run="false">
<inputMatrix>matrix.csv</inputMatrix><outputDir>output</outputDir>
<targetColumn>aki_label</targetColumn><features><column>{feature}</column></features>
<split trainFraction="0.8" seed="2026"/><options>{options}</options>
</model></models></pipelineConfiguration>''',
        encoding="utf8",
    )
    return path


def test_xml_accepts_negative_finite_rosenblatt_threshold_like_the_course(tmp_path):
    path = _xml(
        tmp_path,
        kind="rosenblatt_perceptron",
        options='<option name="threshold" value="-1"/>',
    )
    step = load_config(path).models[0]
    assert step.options["threshold"] == -1.0
    assert step.options["clamped"] is False


@pytest.mark.parametrize(
    "value,expected", [("true", True), ("false", False), ("1", True), ("0", False)]
)
def test_xml_rosenblatt_clamped_is_an_explicit_boolean(tmp_path, value, expected):
    step = load_config(
        _xml(
            tmp_path,
            kind="rosenblatt_perceptron",
            options=f'<option name="clamped" value="{value}"/>',
        )
    ).models[0]
    assert step.options["clamped"] is expected


@pytest.mark.parametrize(
    "options",
    [
        '<option name="clamped" value="yes"/>',
        '<option name="learning_rate" value="0.1"/>',
    ],
)
def test_xml_rejects_nonreference_rosenblatt_options(tmp_path, options):
    with pytest.raises(ValueError, match="clamped|learning_rate"):
        load_config(_xml(tmp_path, kind="rosenblatt_perceptron", options=options))


@pytest.mark.parametrize(
    "feature",
    [
        "patient_id",
        "aki_stage",
        "label_status",
        "label_reason",
        "postoperative/creatinine",
    ],
)
def test_builtin_xml_models_reject_outcome_features_before_execution(tmp_path, feature):
    with pytest.raises(ValueError, match="predictor|feature"):
        load_config(_xml(tmp_path, feature=feature))


@pytest.mark.parametrize(
    "activation", ["relu", "leaky_relu", "sigmoid", "tanh", "linear"]
)
def test_xml_and_cli_share_mlp_hidden_activation_choices(tmp_path, activation):
    path = _xml(tmp_path, options=f'<option name="activation" value="{activation}"/>')
    config = load_config(path)
    assert config.models[0].options["activation"] == activation
    args = build_parser().parse_args(["mlp-train", "--activation", activation])
    assert args.activation == activation
