import csv
from pathlib import Path

from src.models.evaluation.report import save_classification_report
from src.models.rosenblatt_perceptron.plots.charts import save_validation_plots


def save_report(path: Path, patient_ids, y_true, y_pred, scores):
    if len(patient_ids) != len(y_true) or len(scores) != len(y_true):
        raise ValueError("Report rows must be aligned")
    path = Path(path)
    report = save_classification_report(
        path,
        y_true,
        y_pred,
        labels=[-1, 1],
        scores=scores,
        label_names=["No AKI (-1)", "AKI (+1)"],
    )
    with (path / "predictions.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "true_label", "predicted_label", "decision_score"])
        w.writerows(zip(patient_ids, y_true, y_pred, scores))
    save_validation_plots(y_true, scores, report, path / "plots")
    return report
