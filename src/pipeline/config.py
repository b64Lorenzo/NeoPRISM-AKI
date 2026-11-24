"""Strict, side-effect-free XML configuration loading.

Domain INIs remain owned by the existing preprocessing commands.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol

if TYPE_CHECKING:
    from src.pipeline.registry import ModelRegistry

MAX_XML_BYTES = 1024 * 1024
SCHEMA_HINT = "{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation"


@dataclass(frozen=True)
class ExtractionConfig:
    config: Path
    output_dir: Path
    inputs: tuple[Path, ...]
    run: bool = False


@dataclass(frozen=True)
class PlottingConfig:
    config: Path
    run: bool = False


@dataclass(frozen=True)
class InputsModelConfig:
    config: Path
    unlabeled: bool = False
    run: bool = False


@dataclass(frozen=True)
class PreprocessingConfig:
    extraction: ExtractionConfig | None = None
    plotting: PlottingConfig | None = None
    inputs_model: InputsModelConfig | None = None


@dataclass(frozen=True)
class ModelConfig:
    id: str
    model_type: str
    input_matrix: Path
    output_dir: Path
    target_column: str
    features: tuple[str, ...]
    train_fraction: float
    seed: int
    run: bool = False


class PreparedModel(Protocol):
    """A registry-bound model whose options were validated before execution."""

    @property
    def config(self) -> ModelConfig: ...

    def execute(self) -> int | None: ...


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    preprocessing: PreprocessingConfig | None = None
    models: tuple[PreparedModel, ...] = ()


class _NoDTDTreeBuilder(ET.TreeBuilder):
    def doctype(self, name: str, pubid: str | None, system: str | None) -> NoReturn:
        # A parser callback also catches UTF-16 declarations before expansion.
        raise ValueError("DTD declarations are not allowed in pipeline XML")

    def pi(self, target: str, text: str | None = None) -> NoReturn:
        raise ValueError("Processing instructions are not allowed in pipeline XML")


def _attributes(
    element: ET.Element,
    required: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
) -> None:
    unknown = element.attrib.keys() - set(required) - set(optional)
    if unknown:
        raise ValueError(f"Unknown attribute on <{element.tag}>: {min(unknown)}")
    for name in required:
        _attribute(element, name)


def _attribute(element: ET.Element, name: str, default: str | None = None) -> str:
    value = element.get(name, default)
    if value is None or not value.strip():
        raise ValueError(f"<{element.tag}> requires a nonempty {name} attribute")
    return value.strip()


def _children(
    element: ET.Element,
    order: tuple[str, ...] = (),
    repeated: tuple[str, ...] = (),
) -> dict[str, list[ET.Element]]:
    if element.text and element.text.strip():
        raise ValueError(f"Unexpected text in <{element.tag}>")
    children: dict[str, list[ET.Element]] = {name: [] for name in order}
    last = -1
    for child in element:
        if child.tag not in children:
            raise ValueError(f"Unknown element <{child.tag}> in <{element.tag}>")
        index = order.index(child.tag)
        if index < last:
            raise ValueError(f"Elements out of order in <{element.tag}>")
        if children[child.tag] and child.tag not in repeated:
            raise ValueError(f"Duplicate <{child.tag}> in <{element.tag}>")
        if child.tail and child.tail.strip():
            raise ValueError(f"Unexpected text after <{child.tag}>")
        children[child.tag].append(child)
        last = index
    return children


def _text(element: ET.Element) -> str:
    _attributes(element)
    if len(element):
        raise ValueError(f"<{element.tag}> must contain text only")
    value = (element.text or "").strip()
    if not value:
        raise ValueError(f"<{element.tag}> must not be empty")
    return value


def _boolean(element: ET.Element, name: str, default: str = "false") -> bool:
    value = _attribute(element, name, default)
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"<{element.tag}> {name} must be true, false, 1 or 0")
    return value in {"true", "1"}


def _path(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def parse_integer(value: str, name: str, *, minimum: int = 0) -> int:
    """Parse the complete decimal token, without Python's underscore extension."""
    if not re.fullmatch(r"[+-]?[0-9]+", value):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def parse_number(
    value: str,
    name: str,
    *,
    minimum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    """Parse a finite XML numeric token, optionally enforcing a lower bound."""
    if not re.fullmatch(
        r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value
    ):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and (
        number < minimum or (exclusive_minimum and number == minimum)
    ):
        comparison = "greater than" if exclusive_minimum else "at least"
        raise ValueError(f"{name} must be {comparison} {minimum}")
    return number


def _preprocessing(element: ET.Element, base: Path) -> PreprocessingConfig:
    _attributes(element)
    children = _children(element, ("extraction", "plotting", "inputsModel"))
    extraction = None
    plotting = None
    inputs_model = None
    if children["extraction"]:
        node = children["extraction"][0]
        _attributes(node, ("config", "outputDir"), ("run",))
        inputs = _children(node, ("input",), ("input",))["input"]
        if not inputs:
            raise ValueError("<extraction> requires at least one <input>")
        extraction = ExtractionConfig(
            config=_path(_attribute(node, "config"), base),
            output_dir=_path(_attribute(node, "outputDir"), base),
            inputs=tuple(_path(_text(item), base) for item in inputs),
            run=_boolean(node, "run"),
        )
    if children["plotting"]:
        node = children["plotting"][0]
        _attributes(node, ("config",), ("run",))
        _children(node)
        plotting = PlottingConfig(
            config=_path(_attribute(node, "config"), base), run=_boolean(node, "run")
        )
    if children["inputsModel"]:
        node = children["inputsModel"][0]
        _attributes(node, ("config",), ("run", "unlabeled"))
        _children(node)
        inputs_model = InputsModelConfig(
            config=_path(_attribute(node, "config"), base),
            unlabeled=_boolean(node, "unlabeled"),
            run=_boolean(node, "run"),
        )
    return PreprocessingConfig(extraction, plotting, inputs_model)


def _model(element: ET.Element, base: Path) -> tuple[ModelConfig, dict[str, str]]:
    _attributes(element, ("id", "type"), ("run",))
    identifier = _attribute(element, "id")
    model_type = _attribute(element, "type")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", identifier):
        raise ValueError("Model id must match [A-Za-z_][A-Za-z0-9_.-]*")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", model_type):
        raise ValueError("Model type must match [a-z][a-z0-9_]*")
    order = ("inputMatrix", "outputDir", "targetColumn", "features", "split", "options")
    children = _children(element, order)
    for name in order:
        if not children[name]:
            raise ValueError(f"Model {identifier} requires <{name}>")
    features_node = children["features"][0]
    _attributes(features_node)
    features = tuple(
        _text(node)
        for node in _children(features_node, ("column",), ("column",))["column"]
    )
    if not features:
        raise ValueError(f"Model {identifier} requires at least one feature column")
    if len(set(features)) != len(features):
        raise ValueError(f"Model {identifier} has duplicate feature columns")
    target = _text(children["targetColumn"][0])
    if target in features:
        raise ValueError(f"Model {identifier} target column must not be a feature")
    split = children["split"][0]
    _attributes(split, ("trainFraction", "seed"))
    _children(split)
    fraction = parse_number(
        _attribute(split, "trainFraction"),
        "trainFraction",
        minimum=0,
        exclusive_minimum=True,
    )
    if fraction >= 1:
        raise ValueError("trainFraction must be strictly between 0 and 1")
    options_node = children["options"][0]
    _attributes(options_node)
    options: dict[str, str] = {}
    for node in _children(options_node, ("option",), ("option",))["option"]:
        _attributes(node, ("name", "value"))
        _children(node)
        name = _attribute(node, "name")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Model option name must match [a-z][a-z0-9_]*")
        if name in options:
            raise ValueError(f"Duplicate option {name} on model {identifier}")
        options[name] = _attribute(node, "value")
    config = ModelConfig(
        id=identifier,
        model_type=model_type,
        input_matrix=_path(_text(children["inputMatrix"][0]), base),
        output_dir=_path(_text(children["outputDir"][0]), base),
        target_column=target,
        features=features,
        train_fraction=fraction,
        seed=parse_integer(_attribute(split, "seed"), "seed"),
        run=_boolean(element, "run"),
    )
    return config, options


def _models(
    element: ET.Element, base: Path, registry: ModelRegistry
) -> tuple[PreparedModel, ...]:
    _attributes(element)
    nodes = _children(element, ("model",), ("model",))["model"]
    models: list[PreparedModel] = []
    identifiers: set[str] = set()
    enabled_outputs: set[Path] = set()
    for node in nodes:
        config, options = _model(node, base)
        if config.id in identifiers:
            raise ValueError(f"Duplicate model id: {config.id}")
        identifiers.add(config.id)
        if config.run:
            if config.output_dir in enabled_outputs:
                raise ValueError("Enabled models must not share an output directory")
            enabled_outputs.add(config.output_dir)
        models.append(registry.prepare(config, options))
    return tuple(models)


def load_config(
    config_path: str | Path,
    registry: ModelRegistry | None = None,
    *,
    max_bytes: int = MAX_XML_BYTES,
) -> PipelineConfig:
    """Parse and validate without reading referenced INIs or matrix files."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("XML size limit must be a positive integer")
    path = Path(config_path).resolve()
    with path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"Pipeline XML exceeds the {max_bytes}-byte size limit")
    try:
        root = ET.fromstring(payload, parser=ET.XMLParser(target=_NoDTDTreeBuilder()))
    except ET.ParseError as error:
        raise ValueError(f"Invalid pipeline XML: {error}") from error
    if root.tag != "pipelineConfiguration":
        raise ValueError(
            "Root element must be <pipelineConfiguration> without a namespace"
        )
    _attributes(root, optional=(SCHEMA_HINT,))
    children = _children(root, ("preprocessing", "models"))
    preprocessing = (
        _preprocessing(children["preprocessing"][0], path.parent)
        if children["preprocessing"]
        else None
    )
    models: tuple[PreparedModel, ...] = ()
    if children["models"]:
        if registry is None:
            from src.pipeline.registry import create_default_registry

            registry = create_default_registry()
        models = _models(children["models"][0], path.parent, registry)
    return PipelineConfig(path, preprocessing, models)
