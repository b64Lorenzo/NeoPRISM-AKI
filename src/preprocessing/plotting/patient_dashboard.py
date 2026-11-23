"""Per-patient HTML dashboards with perioperative events and operation boundaries.

The input is phase-separated extractor CSVs. Dates are deliberately not used;
all charts and clinical event rows are expressed in operation-relative minutes.
Each HTML contains only one patient's data.
"""
from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path
import math
import re

import plotly.graph_objects as go

from src.preprocessing.plotting.data import Observation
from src.preprocessing.plotting.selection import ParameterInfo

PHASES = ("preoperative", "intraoperative", "postoperative")
PHASE_LABEL = {"preoperative": "Preoperative", "intraoperative": "Intraoperative",
               "postoperative": "Postoperative"}
PHASE_COLOR = {"preoperative": "#62a1f3", "intraoperative": "#f3ad49",
               "postoperative": "#4dc7aa"}


def _cell(value: object) -> str:
    return escape(str(value), quote=True)


def _minute(value: float | None) -> str:
    return f"{value:,.2f} min" if value is not None else "Unavailable"


def _safe_filename(patient: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9_.-]", "_", patient).strip("._")
    if not filename or filename in {".", ".."}:
        raise ValueError(f"Invalid patient ID: {patient!r}")
    return filename


def _table(rows: list[Observation], catalog: dict[str, ParameterInfo]) -> str:
    if not rows:
        return '<p class="muted">No extracted observations available.</p>'
    out = ['<div class="scroll"><table><thead><tr><th>Time (min)</th><th>Parameter</th>'
           '<th>Value</th><th>Unit</th><th>Source field</th><th>Origin</th></tr></thead><tbody>']
    for row in sorted(rows, key=lambda r: (r.timestamp is None,
                                           r.timestamp if r.timestamp is not None else 0,
                                           r.parameter_id)):
        meta = catalog.get(row.key)
        cells = (
            f"{row.timestamp:g}" if row.timestamp is not None else "Untimed",
            row.parameter_id, row.value, meta.unit if meta else "",
            row.source_field, row.value_origin,
        )
        out.append("<tr>" + "".join(f"<td>{_cell(value)}</td>" for value in cells) + "</tr>")
    return "".join(out) + "</tbody></table></div>"


def _event_table(events: list[Observation]) -> str:
    if not events:
        return '<p class="muted">No timestamped Epic events were exported for this patient.</p>'
    out = ['<div class="scroll"><table><thead><tr><th>Time (min)</th><th>Phase</th>'
           '<th>Event</th></tr></thead><tbody>']
    for event in events:
        out.append("<tr>" + "".join(f"<td>{_cell(value)}</td>" for value in (
            f"{event.timestamp:g}" if event.timestamp is not None else "—",
            PHASE_LABEL[event.phase], event.value)) + "</tr>")
    return "".join(out) + "</tbody></table></div>"


def _plot(rows: list[Observation], events: list[Observation],
          end: float | None, theme: str, embed_js: bool, start_known: bool) -> str:
    """Normalize numeric series independently and plot events on a separate lane.

    Normalization is visualization-only: measured values remain unchanged in
    CSVs, tables and hover details. No fitted statistics enter the pipeline.
    """
    dark = theme == "dark"
    figure = go.Figure()
    series: dict[str, list[Observation]] = defaultdict(list)
    for row in rows:
        if row.timestamp is not None and row.numeric_value is not None:
            series[row.key].append(row)

    for key, observations in sorted(series.items()):
        observations.sort(key=lambda row: row.timestamp)
        values = [row.numeric_value for row in observations]
        low, high = min(values), max(values)
        normalized = [0.5 if high == low else (x - low)/(high-low) for x in values]
        name = escape(f"{observations[0].parameter_id} · {PHASE_LABEL[observations[0].phase]}", quote=True)
        figure.add_trace(go.Scatter(
            x=[row.timestamp for row in observations], y=normalized,
            customdata=[[row.numeric_value, row.value_origin] for row in observations],
            name=name, legendgroup=key, mode="lines+markers",
            line={"width": 2, "color": PHASE_COLOR[observations[0].phase]},
            marker={"size": 6},
            hovertemplate=("<b>%{fullData.name}</b><br>Time: %{x:.2f} min"
                           "<br>Measured value: %{customdata[0]:g}"
                           "<br>Normalized (display only): %{y:.3f}"
                           "<br>Origin: %{customdata[1]}<extra></extra>"),
        ))

    if events:
        figure.add_trace(go.Scatter(
            x=[event.timestamp for event in events], y=[-0.20]*len(events),
            customdata=[[escape(event.value, quote=True), PHASE_LABEL[event.phase]] for event in events],
            mode="markers", name="Epic events",
            marker={"symbol": "diamond", "size": 8, "color": "#d084f5", "opacity": .8},
            hovertemplate=("<b>%{customdata[0]}</b><br>Phase: %{customdata[1]}"
                           "<br>Time: %{x:.2f} min<extra>Epic event</extra>"),
        ))

    if start_known:
        figure.add_vline(x=0, line_color="#e3a548", line_dash="dash",
                         annotation_text="Operation start · 0 min", annotation_position="top left")
    if end is not None and end >= 0 and math.isfinite(end):
        figure.add_vline(x=end, line_color="#4ac7a2", line_dash="dash",
                         annotation_text=f"Operation end · {end:g} min",
                         annotation_position="top right")
    figure.update_layout(
        title="Clinical timeline · relative to operation start",
        template="plotly_dark" if dark else "plotly_white",
        height=570, hovermode="closest",
        margin={"l": 70, "r": 65, "t": 110, "b": 75},
        legend={"orientation": "h", "y": -0.24},
        xaxis={"title": "Time from operation start (min)", "zeroline": False},
        yaxis={"title": "Per-series normalized value", "range": [-0.29, 1.08],
               "tickvals": [0, .25, .5, .75, 1]},
    )
    return figure.to_html(full_html=False, include_plotlyjs=True if embed_js else "cdn",
                          config={"responsive": True, "displaylogo": False})


def generate_patient_dashboards(
    observations: list[Observation],
    catalog: dict[str, ParameterInfo],
    patients: list[str],
    output_dir: Path,
    theme: str = "dark",
    include_plotlyjs: bool = True,
    *,
    boundaries: dict[str, dict[str, float | None]] | None = None,
    events: dict[str, list[Observation]] | None = None,
) -> int:
    """Generate a separate patient-only HTML file, including operation end."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for row in observations:
        grouped[row.patient_id].append(row)
    safe_names = [_safe_filename(patient) for patient in patients]
    if len(set(safe_names)) != len(safe_names):
        raise ValueError("Patient identifiers collide after sanitization")
    boundaries = boundaries or {}
    events = events or {}
    dark = theme == "dark"
    bg, surface, ink, muted, border = (
        ("#0c111b", "#172132", "#edf2fa", "#a7b3c6", "#35465d") if dark else
        ("#f5f7fb", "#ffffff", "#182435", "#596b7e", "#dce2eb")
    )
    css = """<style>
:root{--bg:%s;--surface:%s;--ink:%s;--muted:%s;--border:%s}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1580px;margin:auto;padding:24px clamp(12px,3vw,38px)}h1{font-size:28px;letter-spacing:-.03em;margin:0 0 3px}h2{font-size:17px;margin:0 0 12px}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:18px 0}
.card,details{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:15px 18px;margin:12px 0}
.card strong{display:block;font-size:19px;margin-top:3px}.card span{color:var(--muted);font-size:12px}summary{cursor:pointer;font-weight:650;font-size:15px}
.scroll{max-height:450px;overflow:auto;margin-top:12px}table{width:100%%;border-collapse:collapse;text-align:left}
th,td{padding:8px 12px;border-bottom:1px solid var(--border);vertical-align:top;overflow-wrap:anywhere}th{position:sticky;top:0;background:var(--surface)}
.pill{display:inline-block;padding:3px 10px;border-radius:22px;border:1px solid var(--border);margin:6px 5px 6px 0;color:var(--muted)}
.notice{padding:10px 14px;border-left:3px solid #e3a548;background:var(--surface);margin:12px 0}
@media(max-width:750px){main{padding:12px}h1{font-size:22px}.card{padding:11px}}
</style>""" % (bg, surface, ink, muted, border)

    for patient, safe_name in zip(patients, safe_names):
        rows = grouped[patient]
        patient_events = events.get(patient, [])
        patient_boundaries = boundaries.get(patient, {})
        start = patient_boundaries.get("operation_start")
        end = patient_boundaries.get("operation_end")
        post_end = patient_boundaries.get("postoperative_end")
        if start is None:
            # A zero origin is meaningful only when the extractor reports start.
            end = None
        untimed = [row for row in rows if row.timestamp is None]
        phase_rows = {phase: [row for row in rows if row.phase == phase] for phase in PHASES}
        stats = (
            ("Observations", str(len(rows))), ("Epic events", str(len(patient_events))),
            ("Parameters", str(len({row.parameter_id for row in rows}))),
            ("Operation start", "0 min" if start is not None else "Unavailable"),
            ("Operation end", _minute(end)),
            ("Operation duration", _minute(end) if end is not None else "Unavailable"),
        )
        cards = "".join(f'<div class="card"><span>{_cell(label)}</span><strong>{_cell(value)}</strong></div>'
                        for label, value in stats)
        warnings = []
        if start is None:
            warnings.append("Operation start is missing in summary.csv; the zero reference cannot be verified.")
        if end is None:
            warnings.append("Operation end is missing in summary.csv; no end marker can be placed.")
        note = "".join(f'<p class="notice">{_cell(message)}</p>' for message in warnings)
        chart = _plot(rows, patient_events, end, theme, include_plotlyjs, start is not None)
        details = "".join(
            f'<details {"open" if phase == "intraoperative" else ""}><summary>'
            f'{PHASE_LABEL[phase]} · {len(phase_rows[phase])} observations</summary>'
            f'{_table(phase_rows[phase], catalog)}</details>' for phase in PHASES
        )
        start_label = "0 min" if start is not None else "Unavailable"
        boundary_str = (f'<span class="pill">Operation start: {_cell(start_label)}</span>'
                        f'<span class="pill">Operation end: {_cell(_minute(end))}</span>'
                        f'<span class="pill">Postoperative end: {_cell(_minute(post_end))}</span>')
        body = (
            f'<main><header><h1>Patient {_cell(patient)}</h1>'
            '<p class="muted">Perioperative clinical dashboard · individual patient · operation-relative minutes only</p>'
            f'</header><div class="cards">{cards}</div>{note}<div>{boundary_str}</div>'
            f'<section class="card"><h2>Operation timeline and measured parameters</h2>{chart}'
            '<p class="muted">Each numeric series is independently normalized for display; hover to inspect original values. '
            'Diamond markers show Epic events. Vertical dashed lines indicate configured operation boundaries.</p></section>'
            f'<details open><summary>Epic event timeline · {len(patient_events)} events</summary>'
            f'{_event_table(patient_events)}</details>'
            f'<details><summary>Untimed patient attributes · {len(untimed)}</summary>'
            f'{_table(untimed, catalog)}</details>'
            f'{details}<p class="muted">No calendar dates or cohort plots are included. '
            'Operation boundaries are imported from extraction summary.csv, never inferred from the last chart point. '
            'Missing phases or boundaries indicate unavailable evidence, not a clinical finding.</p></main>'
        )
        html = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Patient {_cell(patient)}</title>{css}</head><body>{body}</body></html>'
        )
        (output_dir / f"{safe_name}.html").write_text(html, encoding="utf-8")
    return len(patients)
