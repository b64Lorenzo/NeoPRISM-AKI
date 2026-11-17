"""Parse the root XML document with optional preprocessing and model stages."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.logging.config import LoggingConfig
from src.models.config import ModelsConfig
from src.preprocessing.config import PreprocessingConfig
from src.utils.config import XmlConfig
from src.utils.utils import _section


@dataclass(frozen=True)
class PipelineConfig(XmlConfig):
    project_root: Path
    logging: LoggingConfig
    preprocessing: PreprocessingConfig | None
    models: ModelsConfig

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        config_path = Path(path).resolve()
        try:
            root = ET.parse(config_path).getroot()
        except (ET.ParseError, OSError) as error:
            raise ValueError(f"Unable to read pipeline XML {config_path}: {error}") from error
        if root.tag != "pipelineConfiguration" or root.get("version") != "2":
            raise ValueError("Pipeline XML must have a pipelineConfiguration root with version='2'")
        project_root = config_path.parent.parent
        preprocessing_sections = root.findall("preprocessing")
        model_sections = root.findall("models")
        if len(preprocessing_sections) > 1:
            raise ValueError("Duplicate XML section: preprocessing")
        if len(model_sections) > 1:
            raise ValueError("Duplicate XML section: models")
        return cls(
            project_root,
            LoggingConfig.from_xml(_section(root, "logging"), project_root),
            PreprocessingConfig.from_xml(preprocessing_sections[0], project_root)
            if preprocessing_sections else None,
            ModelsConfig.from_xml(model_sections[0] if model_sections else None, project_root),
        )

    def as_dict(self) -> dict:
        result = super().as_dict()
        result["models"] = self.models.as_dict()["models"]
        return result


def load_pipeline_config(path: str | Path) -> dict:
    return PipelineConfig.load(path).as_dict()
