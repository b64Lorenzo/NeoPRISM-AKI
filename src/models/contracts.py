"""Shared lightweight contracts; importing these never loads a training backend."""

TARGETS = {"aki_label": (-1, 1), "aki_stage": (0, 1, 2, 3)}
OUTCOME_COLUMNS = frozenset(
    {"patient_id", "aki_label", "aki_stage", "label_status", "label_reason"}
)
HIDDEN_ACTIVATIONS = ("relu", "leaky_relu", "sigmoid", "tanh", "linear")


def is_forbidden_predictor(column):
    return column in OUTCOME_COLUMNS or column.startswith("postoperative/")
