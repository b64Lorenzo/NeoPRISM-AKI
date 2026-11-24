"""Shared patient matrices, patient-disjoint splits and training-only transforms."""

import csv
import random
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import mean, median, pstdev

from src.models.contracts import OUTCOME_COLUMNS, TARGETS, is_forbidden_predictor

UNKNOWN = {"", "unknown", "undetermined", "na", "n/a", "none", "null", "?", "nan"}


def read_matrix(path, *, target_column="aki_label", features=None, require_target=True):
    if target_column not in TARGETS:
        raise ValueError("target_column must be aki_label or aki_stage")
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or any(not key for key in fields):
            raise ValueError("Matrix headers must be unique and nonempty")
        if "patient_id" not in fields or (
            require_target and target_column not in fields
        ):
            raise ValueError(
                "Matrix requires patient_id and the selected target column"
            )
        selected = (
            list(features)
            if features is not None
            else [key for key in fields if key not in OUTCOME_COLUMNS]
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Features must be nonempty and distinct")
        if any(is_forbidden_predictor(key) for key in selected):
            raise ValueError(
                "Patient IDs and postoperative/outcome columns cannot be predictors"
            )
        if not set(selected).issubset(fields):
            raise ValueError("Feature columns are missing from the matrix")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed matrix row width")
    ids = [row["patient_id"] for row in rows]
    if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Patient IDs must be unique and nonempty")
    return rows, selected


def split_rows(rows, fraction, seed, *, target_column="aki_label"):
    if target_column not in TARGETS:
        raise ValueError("target_column must be aki_label or aki_stage")
    if not isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("train_fraction must be in (0,1)")
    groups = {label: [] for label in TARGETS[target_column]}
    excluded = 0
    for row in rows:
        raw = (row.get(target_column) or "").strip()
        if raw.casefold() in UNKNOWN:
            excluded += 1
            continue
        if raw not in {str(label) for label in groups} | (
            {"+1"} if target_column == "aki_label" else set()
        ):
            raise ValueError("Invalid target label for the selected target column")
        groups[int(raw)].append(row)
    populated = [group for group in groups.values() if group]
    if len(populated) < 2:
        raise ValueError("Training requires at least two observed classes")
    rng = random.Random(seed)
    training, validation = [], []
    for group in populated:
        if len(group) < 2:
            raise ValueError(
                "Need >=2 patients in each observed class for stratified validation"
            )
        group = group[:]
        rng.shuffle(group)
        count = max(1, min(len(group) - 1, round(len(group) * (1 - fraction))))
        validation.extend(group[:count])
        training.extend(group[count:])
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation, excluded


@dataclass
class Transformer:
    """Training-only numeric median/scaling and categorical one-hot encoding."""

    features: list[str]
    specs: list[dict]

    @classmethod
    def fit(cls, rows, features):
        if not rows:
            raise ValueError("No training rows")
        specs = []
        for key in features:
            observed = [
                row.get(key, "").strip() for row in rows if row.get(key, "").strip()
            ]
            if not observed:
                raise ValueError(f"All training values missing for {key}")
            try:
                values = [float(value) for value in observed]
            except ValueError:
                specs.append(
                    {"type": "categorical", "categories": sorted(set(observed))}
                )
                continue
            if not all(isfinite(value) for value in values):
                raise ValueError(f"Non-finite training value for {key}")
            specs.append(
                {
                    "type": "numeric",
                    "median": median(values),
                    "mean": mean(values),
                    "scale": pstdev(values) or 1.0,
                }
            )
        return cls(list(features), specs)

    def transform_one(self, row):
        result = []
        for key, spec in zip(self.features, self.specs):
            raw = (row.get(key) or "").strip()
            if spec["type"] == "numeric":
                try:
                    value = float(raw) if raw else spec["median"]
                except ValueError:
                    raise ValueError(f"Non-numeric value for {key}") from None
                if not isfinite(value):
                    raise ValueError(f"Non-finite value for {key}")
                result.extend([(value - spec["mean"]) / spec["scale"], float(not raw)])
            else:
                result.extend(float(raw == category) for category in spec["categories"])
                result.append(float(not raw))
        return result

    def transform(self, rows):
        return [self.transform_one(row) for row in rows]

    def to_dict(self):
        return {"features": self.features, "specs": self.specs}

    @classmethod
    def from_dict(cls, data):
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("features"), list)
            or not isinstance(data.get("specs"), list)
        ):
            raise TypeError("Invalid transformer schema")
        features, specs = data["features"], data["specs"]
        if (
            not features
            or any(not isinstance(key, str) or not key for key in features)
            or len(set(features)) != len(features)
        ):
            raise ValueError("Invalid transformer features")
        if len(features) != len(specs):
            raise ValueError("Transformer schema mismatch")
        for spec in specs:
            if not isinstance(spec, dict):
                raise TypeError("Invalid transformer specification")
            if spec.get("type") == "numeric":
                values = [spec.get(key) for key in ("median", "mean", "scale")]
                if (
                    any(
                        not isinstance(value, (int, float)) or not isfinite(value)
                        for value in values
                    )
                    or values[-1] <= 0
                ):
                    raise ValueError("Invalid transformer numeric statistics")
            elif spec.get("type") == "categorical":
                categories = spec.get("categories")
                if (
                    not isinstance(categories, list)
                    or not categories
                    or any(not isinstance(value, str) for value in categories)
                    or len(set(categories)) != len(categories)
                ):
                    raise ValueError("Invalid transformer categories")
            else:
                raise ValueError("Invalid transformer feature type")
        return cls(features, specs)
