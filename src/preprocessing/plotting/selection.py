"""Phase-aware parameter catalog and configurable selection."""
from __future__ import annotations
from dataclasses import dataclass
from src.preprocessing.plotting.config import PlotConfig

@dataclass(frozen=True)
class ParameterInfo:
    id: str
    phase: str
    type: str
    unit: str
    source_fields: tuple[str, ...]
    class_names: tuple[str, ...]
    families: tuple[str, ...]


def build_catalog(parameters: tuple, config: PlotConfig) -> dict[str, ParameterInfo]:
    known = {p.id for p in parameters}
    for family, members in config.families.items():
        unknown = set(members) - known
        if unknown:
            raise ValueError(f"Unknown parameter IDs in family {family}: {sorted(unknown)}")
    catalog = {}
    for p in parameters:
        classes = set()
        for field in p.source_fields:
            parts = field.split('.')
            if len(parts) >= 3 and parts[0].casefold() == 'epic':
                classes.add('.'.join(parts[:2]))
        families = tuple(name for name, members in config.families.items() if p.id in members)
        # Support the new multi-phase parameter schema and legacy single-phase definitions.
        phases = getattr(p, 'phases', None)
        if phases is None:
            phases = tuple(x.strip() for x in p.phase.split(',') if x.strip())
        for phase in phases:
            key = f'{phase}/{p.id}'
            if key in catalog:
                raise ValueError(f'Duplicate parameter-phase combination: {key}')
            catalog[key] = ParameterInfo(
                id=p.id, phase=phase, type=p.type, unit=p.unit,
                source_fields=tuple(p.source_fields),
                class_names=tuple(sorted(classes)), families=families,
            )
    return catalog


def select_parameters(catalog: dict[str, ParameterInfo], config: PlotConfig) -> dict[str, ParameterInfo]:
    s = config.selection
    known_ids = {p.id for p in catalog.values()}
    known_classes = {name for p in catalog.values() for name in p.class_names}
    for label, requested, available in (
        ('parameter', s.parameter_ids, known_ids),
        ('class', s.class_names, known_classes),
        ('family', s.families, set(config.families)),
    ):
        unknown = set(requested) - available
        if unknown:
            raise ValueError(f'Unknown {label} selection: {sorted(unknown)}')
    selected = {}
    for key, p in catalog.items():
        if p.phase not in s.phases:
            continue
        matches = []
        if s.parameter_ids:
            matches.append(p.id in s.parameter_ids)
        if s.class_names:
            matches.append(bool(set(p.class_names) & set(s.class_names)))
        if s.families:
            matches.append(bool(set(p.families) & set(s.families)))
        if not matches or (all(matches) if s.mode == 'intersection' else any(matches)):
            selected[key] = p
    if not selected:
        raise ValueError('No parameters match the plot selection')
    return selected


def select_heatmap_parameters(
    catalog: dict[str, ParameterInfo],
    phases: tuple[str, ...],
    requested: tuple[str, ...],
    *,
    numeric_only: bool = False,
    label: str = "heatmap",
) -> dict[str, ParameterInfo]:
    """Select heatmap columns independently from general plot selection.

    IDs apply to every enabled phase of a parameter. Missingness includes
    categorical parameters; value heatmaps require numeric observations.
    """
    available = {info.id for info in catalog.values()}
    if not requested:
        raise ValueError(f"{label}: parameter_ids cannot be empty; use 'all'")
    if "all" in requested and len(requested) != 1:
        raise ValueError(f"{label}: use 'all' alone or explicit IDs")
    unknown = set(requested) - {"all"} - available
    if unknown:
        raise ValueError(f"{label}: unknown parameter IDs: {sorted(unknown)}")
    selected = {
        key: info for key, info in catalog.items()
        if info.phase in phases
        and ("all" in requested or info.id in requested)
        and (not numeric_only or info.type in {"numeric", "timeseries"})
    }
    if not selected:
        raise ValueError(f"{label}: no eligible parameters in selected phases")
    return selected
