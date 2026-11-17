"""Supervised training inputs, outputs and holdout settings."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import XmlConfig
from src.utils.utils import _ids, _path, _required


@dataclass(frozen=True)
class TrainingFeatureConfig(XmlConfig):
    """Feature schema owned by one model's clean training dataset."""

    id: str
    value_type: str

    @classmethod
    def from_xml(cls, element: ET.Element) -> TrainingFeatureConfig:
        feature_id = element.get("id", "").strip()
        value_type = element.get("type", "").strip()
        if not feature_id:
            raise ValueError("Training features require a nonempty id")
        if value_type not in {"numeric", "categorical"}:
            raise ValueError(f"Unsupported training feature type: {value_type}")
        return cls(feature_id, value_type)

    def as_parameter(self) -> dict[str, str]:
        return {"id": self.id, "parameter_type": self.value_type}


@dataclass(frozen=True)
class TrainingValidationConfig(XmlConfig):
    """Supervised training inputs, outputs and holdout settings."""

    vectors: Path
    outcomes: Path
    training_set: Path
    validation_set: Path
    model_output: Path
    train_fraction: float
    epochs: int
    seed: int
    omit: list[str]
    features: tuple[TrainingFeatureConfig, ...]

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> TrainingValidationConfig:
        fraction = float(_required(element, "trainFraction"))
        epochs = int(_required(element, "epochs"))
        if not 0 < fraction < 1 or epochs < 1:
            raise ValueError("Training requires trainFraction between 0 and 1 and positive epochs")
        feature_list = element.find("features")
        if feature_list is None:
            raise ValueError("Training validation requires a model-owned features section")
        features = tuple(TrainingFeatureConfig.from_xml(item) for item in feature_list.findall("feature"))
        if not features:
            raise ValueError("Training validation requires at least one feature")
        feature_ids = [feature.id for feature in features]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("Training feature IDs must be unique")
        return cls(
            vectors=_path(project_root, _required(element, "vectors")),
            outcomes=_path(project_root, _required(element, "outcomes")),
            training_set=_path(project_root, _required(element, "trainingSet")),
            validation_set=_path(project_root, _required(element, "validationSet")),
            model_output=_path(project_root, _required(element, "modelOutput")),
            train_fraction=fraction,
            epochs=epochs,
            seed=int(_required(element, "seed")),
            omit=_ids(element, "omit"),
            features=features,
        )
