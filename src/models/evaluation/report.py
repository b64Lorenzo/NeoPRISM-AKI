"""Reusable CSV, JSON and self-contained interactive confusion-matrix reports."""

import csv
import json
from pathlib import Path

from .metrics import classification_metrics, evaluate_binary


def save_classification_report(
    path, y_true, y_pred, *, labels, scores=None, label_names=None
):
    labels = list(labels)
    report = (
        evaluate_binary(y_true, y_pred, scores)
        if labels == [-1, 1]
        else classification_metrics(y_true, y_pred, labels=labels)
    )
    names = list(label_names) if label_names is not None else [str(v) for v in labels]
    if len(names) != len(labels):
        raise ValueError("Label names must match label order")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf8"
    )
    with (path / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *names])
        writer.writerows(
            [name, *row] for name, row in zip(names, report["confusion_matrix"])
        )

    import plotly.graph_objects as go

    figure = go.Figure(
        go.Heatmap(
            z=report["confusion_matrix"],
            x=names,
            y=names,
            colorscale="Blues",
            text=report["confusion_matrix"],
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Confusion matrix",
        xaxis_title="Predicted label",
        yaxis_title="True label",
        template="plotly_dark",
    )
    figure.update_yaxes(autorange="reversed")
    figure.write_html(path / "confusion_matrix.html", include_plotlyjs=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return report
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(report["confusion_matrix"], cmap="Blues")
    for i, row in enumerate(report["confusion_matrix"]):
        for j, count in enumerate(row):
            ax.text(j, i, str(count), ha="center", va="center")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(path / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    return report
