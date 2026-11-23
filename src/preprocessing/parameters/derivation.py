"""Extensible, deterministic, patient-local derivations.

Register new functions with @register("method_name"). Each function accepts a
DerivationContext and configured Parameter, and returns a DerivedValue or None.
No statistical fitting or cross-patient information is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.preprocessing.parameters.config import Parameter, ExtractionConfig

WEIGHT_TO_KG = {"kg": 1.0, "g": 0.001, "lb": 0.45359237}
HEIGHT_TO_M = {"m": 1.0, "cm": 0.01, "in": 0.0254}

@dataclass(frozen=True)
class DerivedValue:
    value: float
    source_field: str
    origin: str
    time_minutes: str = ""

@dataclass
class DerivationContext:
    rows: dict[str, list[dict[str, str]]]
    boundaries: dict[str, float | None]

    def last_preoperative(self, identifier: str) -> float | None:
        """Last timestamped pre-op reading, otherwise a unique untimed value."""
        timed: list[tuple[float, float]] = []
        untimed: set[float] = set()
        for row in self.rows["preoperative"]:
            if row["parameter_id"] != identifier:
                continue
            try:
                value = float(row["value"])
                if not math.isfinite(value) or value <= 0:
                    continue
                stamp = row.get("time_minutes", "")
                if stamp:
                    time = float(stamp)
                    if math.isfinite(time) and time < 0:
                        timed.append((time, value))
                else:
                    untimed.add(value)
            except (TypeError, ValueError):
                continue
        if timed:
            return max(timed, key=lambda pair: pair[0])[1]
        return next(iter(untimed)) if len(untimed) == 1 else None

Derivation = Callable[[DerivationContext, "Parameter"], DerivedValue | None]
_REGISTRY: dict[str, Derivation] = {}

def register(name: str):
    """Register a derivation at import time; duplicates are errors."""
    def decorator(func: Derivation) -> Derivation:
        if not name or name in _REGISTRY:
            raise ValueError(f"Duplicate or empty derivation name: {name!r}")
        _REGISTRY[name] = func
        return func
    return decorator

def get_derivation(name: str) -> Derivation:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown derivation {name!r}; available: {sorted(_REGISTRY)}") from exc

@register("bmi")
def bmi(context: DerivationContext, parameter: "Parameter") -> DerivedValue | None:
    weight = context.last_preoperative(parameter.weight_parameter_id)
    height = context.last_preoperative(parameter.height_parameter_id)
    if weight is None or height is None:
        return None
    try:
        kg = weight * WEIGHT_TO_KG[parameter.weight_unit]
        meters = height * HEIGHT_TO_M[parameter.height_unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported BMI unit for {parameter.id}") from exc
    value = kg / (meters * meters) if meters > 0 else float("nan")
    if not math.isfinite(value) or value <= 0:
        return None
    return DerivedValue(value, f"derived:{parameter.weight_parameter_id}+{parameter.height_parameter_id}",
                        "derived_bmi_from_weight_height")

@register("operation_duration")
def operation_duration(context: DerivationContext, parameter: "Parameter") -> DerivedValue | None:
    start = context.boundaries.get("operation_start")
    end = context.boundaries.get("operation_end")
    if start is None or end is None or not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return None
    minutes = (end - start) / 60.0
    unit = parameter.unit.strip().casefold()
    factors = {"minutes": 1, "min": 1, "hours": 60, "h": 60, "days": 1440, "d": 1440}
    if unit not in factors:
        raise ValueError(f"{parameter.id}: unsupported duration unit {unit!r}")
    return DerivedValue(minutes / factors[unit], "derived:operation_start+operation_end",
                        "derived_operation_duration")

def derive_patient_parameters(extracted: dict[str, list[dict[str, str]]],
                              config: "ExtractionConfig",
                              boundaries: dict[str, float | None] | None = None) -> None:
    """Apply configured derivations, preserving observed values and provenance.

    Iterates to support chained derivations; an unresolved rule stays missing.
    Cycles cannot produce values unless they have independent observed inputs.
    """
    context = DerivationContext(extracted, boundaries or {})
    pending = [p for p in config.parameters if p.derive != "none"]
    for _ in range(len(pending)):
        progressed = False
        remaining = []
        for parameter in pending:
            phase = parameter.phases[0]
            if any(row["parameter_id"] == parameter.id for row in extracted[phase]):
                continue
            result = get_derivation(parameter.derive)(context, parameter)
            if result is None:
                remaining.append(parameter)
                continue
            if not math.isfinite(result.value):
                raise ValueError(f"Nonfinite derived result for {parameter.id}")
            extracted[phase].append({
                "time_minutes": result.time_minutes,
                "parameter_id": parameter.id,
                "value": format(result.value, ".12g"),
                "unit": parameter.unit or ("kg/m^2" if parameter.derive == "bmi" else ""),
                "type": parameter.type,
                "source_field": result.source_field,
                "value_origin": result.origin,
            })
            progressed = True
        pending = remaining
        if not progressed:
            break
