"""Phase-aware Plotly charts for extracted clinical observations."""
from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.preprocessing.plotting.config import PlotConfig
from src.preprocessing.plotting.data import Observation, by_parameter, patient_numeric_values
from src.preprocessing.plotting.selection import ParameterInfo


def _slug(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._') or 'plot'


def _save(fig: go.Figure, path: Path, theme: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(template='plotly_dark' if theme == 'dark' else 'plotly_white', margin=dict(l=90, r=40, t=85, b=110))
    fig.write_html(str(path), include_plotlyjs='cdn', full_html=True)


def _filename(p: ParameterInfo) -> str:
    return f'{_slug(p.phase)}_{_slug(p.id)}.html'


def _categories(rows: list[Observation]) -> list[tuple[str, int]]:
    category_patients = defaultdict(set)
    for row in rows:
        category_patients[row.value].add(row.patient_id)
    return sorted(((category, len(patients)) for category, patients in category_patients.items()), key=lambda x: (-x[1], x[0]))


def plot_distributions(observations: list[Observation], catalog: dict[str, ParameterInfo], config: PlotConfig) -> int:
    grouped = by_parameter(observations)
    count = 0
    for key, p in catalog.items():
        rows = grouped.get(key, [])
        if not rows:
            continue
        fig = go.Figure()
        if p.type in {'numeric', 'timeseries'}:
            if config.numeric_aggregation == 'all_observations':
                values = [r.numeric_value for r in rows if r.numeric_value is not None]
                ylabel = 'Observation count'
            else:
                values = list(patient_numeric_values(rows, config.numeric_aggregation).values())
                ylabel = 'Patient count'
            if not values:
                continue
            fig.add_trace(go.Histogram(x=values, nbinsx=config.bins))
            fig.update_xaxes(title_text=f'{p.id} ({p.unit})' if p.unit else p.id)
            fig.update_yaxes(title_text=ylabel)
        else:
            categories = _categories(rows)[:config.max_categories]
            fig.add_trace(go.Bar(x=[x[0] for x in categories], y=[x[1] for x in categories]))
            fig.update_yaxes(title_text='Patient count')
            fig.update_xaxes(tickangle=-45)
        fig.update_layout(title=f'Distribution — {p.id} ({p.phase})')
        _save(fig, config.output_dir / 'distributions' / _filename(p), config.theme)
        count += 1
    return count


def plot_frequencies(observations: list[Observation], catalog: dict[str, ParameterInfo], config: PlotConfig) -> int:
    grouped = by_parameter(observations)
    count = 0
    for key, p in catalog.items():
        if p.type in {'numeric', 'timeseries'}:
            continue
        rows = grouped.get(key, [])
        if not rows:
            continue
        ordered = _categories(rows)[:config.max_categories]
        denominator = len({row.patient_id for row in rows})
        values = [(100 * n / denominator if config.normalize_frequencies else n) for _, n in ordered]
        fig = go.Figure(go.Bar(x=[name for name, _ in ordered], y=values))
        fig.update_layout(title=f'Category frequencies — {p.id} ({p.phase})')
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(title_text='Patients (%)' if config.normalize_frequencies else 'Patient count')
        _save(fig, config.output_dir / 'frequencies' / _filename(p), config.theme)
        count += 1
    return count


def plot_missingness(observations: list[Observation], catalog: dict[str, ParameterInfo], patients: list[str], config: PlotConfig, *, filename: str = "missingness.html", title: str = "Missing data by patient and phase") -> int:
    patients = patients[:config.max_patients]
    if not patients or not catalog:
        return 0
    keys = list(catalog)
    present = {(r.patient_id, r.key) for r in observations}
    matrix = [[0 if (patient, key) in present else 1 for key in keys] for patient in patients]
    fig = go.Figure(go.Heatmap(z=matrix, x=keys, y=patients, zmin=0, zmax=1,
        colorscale=[[0, '#2f855a'], [1, '#c53030']],
        colorbar=dict(title='Missing', tickvals=[0, 1], ticktext=['Present', 'Missing'])))
    fig.update_layout(title=title, height=max(450, min(1800, 24 * len(patients) + 200)))
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(autorange='reversed')
    _save(fig, config.output_dir / 'heatmaps' / filename, config.theme)
    return 1


def plot_value_heatmap(observations: list[Observation], catalog: dict[str, ParameterInfo], patients: list[str], config: PlotConfig) -> int:
    patients = patients[:config.max_patients]
    keys = [key for key, p in catalog.items() if p.type in {'numeric', 'timeseries'}]
    if not patients or not keys:
        return 0
    grouped = by_parameter(observations)
    columns = {}
    original = {}
    for key in keys:
        values = patient_numeric_values(grouped.get(key, []), config.heatmap_aggregation)
        original[key] = values
        if not values:
            columns[key] = {}
            continue
        low, high = min(values.values()), max(values.values())
        columns[key] = {patient: (0.5 if high == low else (value - low) / (high - low)) for patient, value in values.items()}
    matrix = [[columns[key].get(patient) for key in keys] for patient in patients]
    custom = [[[original[key].get(patient)] for key in keys] for patient in patients]
    fig = go.Figure(go.Heatmap(z=matrix, x=keys, y=patients, customdata=custom, zmin=0, zmax=1,
        colorscale='Viridis', colorbar=dict(title='Min-max scaled'), hoverongaps=False,
        hovertemplate='Patient: %{y}<br>Parameter/phase: %{x}<br>Scaled: %{z:.3f}<br>Original: %{customdata[0]}<extra></extra>'))
    fig.update_layout(title='Phase-specific patient parameter values (column-wise scaled)', height=max(450, min(1800, 24 * len(patients) + 200)))
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(autorange='reversed')
    _save(fig, config.output_dir / 'heatmaps' / 'values.html', config.theme)
    return 1


def _subplots(keys: list[str], title: str):
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True, subplot_titles=keys,
        vertical_spacing=min(0.08, 0.4 / max(1, len(keys))))
    fig.update_layout(title=title, height=max(500, 320 * len(keys)))
    return fig


def plot_time_series(observations: list[Observation], catalog: dict[str, ParameterInfo], config: PlotConfig) -> int:
    grouped = by_parameter(observations)
    groups = defaultdict(list)
    for key, p in catalog.items():
        if p.type != 'timeseries':
            continue
        if config.time_series_group_by == 'class':
            names = p.class_names or ('unclassified',)
        elif config.time_series_group_by == 'family':
            names = p.families or ('unclassified',)
        else:
            names = (p.id,)
        for name in names:
            groups[(p.phase, name)].append(key)
    count = 0
    for (phase, name), keys in groups.items():
        keys = list(dict.fromkeys(keys))
        if not any(r.timestamp is not None and r.numeric_value is not None for key in keys for r in grouped.get(key, [])):
            continue
        fig = _subplots(keys, f'Time series — {name} ({phase})')
        for i, key in enumerate(keys, start=1):
            by_patient = defaultdict(list)
            for row in grouped.get(key, []):
                if row.timestamp is not None and row.numeric_value is not None:
                    by_patient[row.patient_id].append(row)
            for patient in sorted(by_patient)[:config.time_series_max_patients]:
                series = sorted(by_patient[patient], key=lambda row: row.timestamp)
                fig.add_trace(go.Scatter(x=[r.timestamp for r in series], y=[r.numeric_value for r in series],
                    mode='lines+markers', name=patient, legendgroup=patient, showlegend=i == 1), row=i, col=1)
            fig.update_yaxes(title_text=catalog[key].unit or 'Value', row=i, col=1)
        fig.update_xaxes(title_text='Elapsed time from operation start (min)', row=len(keys), col=1)
        _save(fig, config.output_dir / 'time_series' / f'{_slug(phase)}_{_slug(name)}.html', config.theme)
        count += 1
    return count


def plot_patient_timelines(observations: list[Observation], catalog: dict[str, ParameterInfo], patients: list[str], config: PlotConfig) -> int:
    grouped = by_parameter(observations)
    count = 0
    for patient in patients[:config.timeline_max_patients]:
        keys = [key for key, p in catalog.items() if p.type in {'numeric', 'timeseries'} and any(
            r.patient_id == patient and r.timestamp is not None and r.numeric_value is not None for r in grouped.get(key, []))]
        if not keys:
            continue
        fig = _subplots(keys, f'Patient timeline — {patient}')
        for i, key in enumerate(keys, start=1):
            series = sorted((r for r in grouped.get(key, []) if r.patient_id == patient and r.timestamp is not None and r.numeric_value is not None), key=lambda r: r.timestamp)
            fig.add_trace(go.Scatter(x=[r.timestamp for r in series], y=[r.numeric_value for r in series],
                mode='lines+markers', name=key, showlegend=False), row=i, col=1)
            fig.update_yaxes(title_text=catalog[key].unit or 'Value', row=i, col=1)
        fig.update_xaxes(title_text='Elapsed time from operation start (min)', row=len(keys), col=1)
        _save(fig, config.output_dir / 'timelines' / f'{_slug(patient)}.html', config.theme)
        count += 1
    return count
