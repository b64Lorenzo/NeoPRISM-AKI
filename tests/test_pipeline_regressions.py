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


def test_xml_rejects_negative_rosenblatt_threshold_during_loading(tmp_path):
    path = _xml(
        tmp_path,
        kind="rosenblatt_perceptron",
        options='<option name="threshold" value="-1"/>',
    )
    with pytest.raises(ValueError, match="threshold"):
        load_config(path)


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
