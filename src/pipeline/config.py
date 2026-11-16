"""Parse the root XML document; each component owns its child settings."""

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
    preprocessing: PreprocessingConfig
    model: ModelsConfig

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        config_path = Path(path).resolve()
        try:
            root = ET.parse(config_path).getroot()
        except (ET.ParseError, OSError) as error:
            raise ValueError(f"Unable to read pipeline XML {config_path}: {error}") from error
        if root.tag != "pipelineConfiguration" or root.get("version") != "1":
            raise ValueError("Pipeline XML must have a pipelineConfiguration root with version='1'")
        project_root = config_path.parent.parent
        return cls(
            project_root,
            LoggingConfig.from_xml(_section(root, "logging"), project_root),
            PreprocessingConfig.from_xml(_section(root, "preprocessing"), project_root),
            ModelsConfig.from_xml(_section(root, "model"), project_root),
        )


def load_pipeline_config(path: str | Path) -> dict:
    return PipelineConfig.load(path).as_dict()
