"""Perceptron-specific XML controls composed from reusable children."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.models.base import ModelConfig
from src.models.novel_data.config import NovelDataConfig
from src.models.plotting.config import PlotConfig
from src.models.training_validation.config import TrainingValidationConfig
from src.utils.utils import _boolean, _required


@dataclass(frozen=True)
class PerceptronModelConfig(ModelConfig):
    """Perceptron execution mode and typed child settings."""

    xml_name = "perceptron"
    run: bool
    mode: str
    training_validation: TrainingValidationConfig
    novel_data: NovelDataConfig
    plotting: PlotConfig

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> PerceptronModelConfig:
        training = element.find("trainingValidation")
        novel = element.find("novelData")
        plotting = element.find("plotting")
        if training is None or novel is None or plotting is None:
            raise ValueError(
                "Perceptron requires trainingValidation, novelData, and plotting sections"
            )
        mode = _required(element, "mode")
        if mode not in {"train-validation", "novel-data"}:
            raise ValueError("Perceptron mode must be train-validation or novel-data")
        model_plot = PlotConfig.from_xml(plotting, project_root)
        if model_plot.run and mode != "train-validation":
            raise ValueError("Perceptron validation plotting requires train-validation mode")
        return cls(
            run=_boolean(element, "run"),
            mode=mode,
            training_validation=TrainingValidationConfig.from_xml(training, project_root),
            novel_data=NovelDataConfig.from_xml(novel, project_root),
            plotting=model_plot,
        )
