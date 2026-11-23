
"""INI configuration for phase-aware parameter extraction."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


PHASES = (
    "preoperative",
    "intraoperative",
    "postoperative",
)

TYPES = {
    "numeric",
    "str",
    "bool",
    "timeseries",
}


def _items(value: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class Parameter:
    id: str
    phases: tuple[str, ...]
    source_fields: tuple[str, ...]
    type: str
    unit: str
    timed: bool
    derive: str = "none"
    weight_parameter_id: str = "weight"
    height_parameter_id: str = "height"
    weight_unit: str = "kg"
    height_unit: str = "cm"
    derive_options: tuple[tuple[str, str], ...] = ()

    @property
    def phase(self) -> str:
        """Compatibility display for code expecting .phase."""
        return ", ".join(self.phases)


@dataclass(frozen=True)
class Boundary:
    fields: tuple[str, ...]
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class ExtractionConfig:
    parameters: tuple[Parameter, ...]
    missing_tokens: frozenset[str]
    strict_columns: bool
    timestamp_fields: tuple[str, ...]
    boundaries: dict[str, Boundary]


def load_config(path: str | Path) -> ExtractionConfig:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Parameter configuration not found: {path}"
        )

    parser = configparser.ConfigParser(
        interpolation=None,
        strict=True,
    )
    parser.optionxform = str

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as stream:
        parser.read_file(stream)

    if "settings" not in parser:
        raise ValueError("Missing [settings] section")

    if "timestamps" not in parser:
        raise ValueError("Missing [timestamps] section")

    missing = frozenset(
        value.casefold()
        for value in _items(
            parser.get(
                "settings",
                "missing_tokens",
                fallback="NA,N/A,null,None,nan",
            )
        )
    ) | {""}

    strict = parser.getboolean(
        "settings",
        "strict_columns",
        fallback=False,
    )

    timestamp_fields = _items(
        parser.get("timestamps", "fields")
    )

    if not timestamp_fields:
        raise ValueError(
            "At least one timestamp field is required"
        )

    boundaries: dict[str, Boundary] = {}

    for name in (
        "operation_start",
        "operation_end",
        "postoperative_end",
    ):
        section = f"boundary.{name}"

        if section not in parser:
            if name == "postoperative_end":
                boundaries[name] = Boundary((), ())
                continue

            raise ValueError(f"Missing [{section}]")

        fields = _items(
            parser.get(section, "fields")
        )
        phrases = _items(
            parser.get(section, "phrases")
        )

        if bool(fields) != bool(phrases):
            raise ValueError(
                f"[{section}] requires both fields and phrases"
            )

        if name != "postoperative_end" and not fields:
            raise ValueError(
                f"[{section}] cannot be empty"
            )

        boundaries[name] = Boundary(
            fields=fields,
            phrases=phrases,
        )

    parameters: list[Parameter] = []
    seen: set[str] = set()

    for section in parser.sections():
        if not section.startswith("parameter."):
            continue

        identifier = section.removeprefix(
            "parameter."
        ).strip()

        if not identifier or identifier in seen:
            raise ValueError(
                f"Invalid or duplicate parameter ID: {identifier}"
            )

        seen.add(identifier)

        phases = tuple(
            phase.casefold()
            for phase in _items(
                parser.get(section, "phase")
            )
        )

        if not phases:
            raise ValueError(
                f"No phases configured for {identifier}"
            )

        invalid_phases = set(phases) - set(PHASES)

        if invalid_phases:
            raise ValueError(
                f"Invalid phases for {identifier}: "
                f"{sorted(invalid_phases)}"
            )

        if len(phases) != len(set(phases)):
            raise ValueError(
                f"Duplicate phases for {identifier}"
            )

        # Store phases in their chronological order.
        ordered_phases = tuple(
            phase
            for phase in PHASES
            if phase in phases
        )

        kind = parser.get(
            section,
            "type",
        ).strip().casefold()

        if kind not in TYPES:
            raise ValueError(
                f"Invalid type for {identifier}: {kind}"
            )

        fields = _items(
            parser.get(section, "source_fields")
        )

        derive = parser.get(section, "derive", fallback="none").strip().casefold()
        if derive != "none":
            from src.preprocessing.parameters.derivation import get_derivation
            get_derivation(derive)
        if not fields and derive == "none":
            raise ValueError(f"No source fields for {identifier}")

        timed = parser.getboolean(
            section,
            "timed",
            fallback=True,
        )

        # Untimed observations cannot be reliably assigned
        # to more than one clinical phase.
        if not timed and len(ordered_phases) > 1:
            raise ValueError(
                f"{identifier}: multiple phases require timed=true"
            )

        # Untimed static values are treated as preoperative
        # patient characteristics.
        if not timed and ordered_phases != ("preoperative",) and derive != "operation_duration":
            raise ValueError(
                f"{identifier}: timed=false is supported "
                "only for preoperative parameters"
            )

        weight_id = parser.get(section, "weight_parameter_id", fallback="weight").strip()
        height_id = parser.get(section, "height_parameter_id", fallback="height").strip()
        weight_unit = parser.get(section, "weight_unit", fallback="kg").strip().casefold()
        height_unit = parser.get(section, "height_unit", fallback="cm").strip().casefold()
        if derive == "bmi":
            if kind != "numeric" or timed or ordered_phases != ("preoperative",):
                raise ValueError(f"{identifier}: BMI must be numeric, untimed, preoperative")
            if weight_unit not in {"kg", "g", "lb"} or height_unit not in {"m", "cm", "in"}:
                raise ValueError(f"{identifier}: unsupported BMI input units")
            if not weight_id or not height_id or weight_id == height_id:
                raise ValueError(f"{identifier}: BMI requires distinct weight/height parameter IDs")
        parameters.append(
            Parameter(
                id=identifier,
                phases=ordered_phases,
                source_fields=fields,
                type=kind,
                unit=parser.get(
                    section,
                    "unit",
                    fallback="",
                ).strip(),
                timed=timed,
                derive=derive,
                weight_parameter_id=weight_id,
                height_parameter_id=height_id,
                weight_unit=weight_unit,
                height_unit=height_unit,
                derive_options=tuple((k, v.strip()) for k, v in parser.items(section)
                                     if k not in {"phase", "source_fields", "type", "unit", "timed", "derive"}),
            )
        )

    if not parameters:
        raise ValueError(
            "No [parameter.*] sections found"
        )

    by_id = {parameter.id: parameter for parameter in parameters}
    for parameter in parameters:
        if parameter.derive == "bmi":
            for dep in (parameter.weight_parameter_id, parameter.height_parameter_id):
                if dep not in by_id:
                    raise ValueError(f"{parameter.id}: missing configured dependency {dep!r}")
                source = by_id[dep]
                if source.type != "numeric" or source.phases != ("preoperative",):
                    raise ValueError(f"{parameter.id}: dependency {dep!r} must be preoperative numeric")
            if parameter.id in (parameter.weight_parameter_id, parameter.height_parameter_id):
                raise ValueError(f"{parameter.id}: BMI cannot depend on itself")
    for parameter in parameters:
        if parameter.derive == "operation_duration":
            if parameter.phases != ("intraoperative",) or parameter.type != "numeric" or parameter.timed:
                raise ValueError(f"{parameter.id}: operation_duration requires intraoperative, numeric, timed=false")
    return ExtractionConfig(
        parameters=tuple(parameters),
        missing_tokens=missing,
        strict_columns=strict,
        timestamp_fields=timestamp_fields,
        boundaries=boundaries,
    )
