"""Render missingness of selected mapped observations across operative phases."""

from pathlib import Path

from src.preprocessing.plotting.heatmaps import sample_patients, write_heatmap


def write_patient_missingness(
    path: Path, patients: list[dict], parameters: list[dict], *, theme: str = "light"
) -> None:
    sample = sample_patients(patients)
    ids = [item["id"] for item in parameters]
    matrix = []
    for patient in sample:
        observations = {
            **patient["features"],
            **patient["duringoperative_measurements"],
            **patient["postoperative_measurements"],
        }
        matrix.append([float(not observations.get(parameter_id, "")) for parameter_id in ids])
    write_heatmap(
        path,
        matrix,
        [patient["patient_id"] for patient in sample],
        ids,
        "Parameter missingness by patient",
        "Missing (1) / present (0)",
        row_axis_label="Patient pseudonym",
        colorscale=[[0, "#2f855a"], [1, "#c53030"]],
        colorbar_tickvals=[0, 1],
        colorbar_ticktext=["Present", "Missing"],
        theme=theme,
    )
