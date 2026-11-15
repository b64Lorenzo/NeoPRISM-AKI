"""Compact, reusable HTML shell for a patient-selectable Plotly chart."""
from __future__ import annotations

import json
from dataclasses import dataclass
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class DashboardLayout:
    chart_height: int = 740
    panel_max_height: int = 320

    def __post_init__(self) -> None:
        if not 320 <= self.chart_height <= 900:
            raise ValueError("chartHeight must be between 320 and 900 pixels")
        if not 120 <= self.panel_max_height <= 900:
            raise ValueError("panelMaxHeight must be between 120 and 900 pixels")

    @classmethod
    def from_xml(cls, element: ET.Element | None) -> DashboardLayout:
        if element is None:
            raise ValueError("Missing patientTimeline/layout section")
        values = []
        for name in ("chartHeight", "panelMaxHeight"):
            node = element.find(name)
            if node is None or node.text is None:
                raise ValueError(f"Missing layout/{name}")
            try:
                values.append(int(node.text.strip()))
            except ValueError as error:
                raise ValueError(f"layout/{name} must be an integer") from error
        return cls(*values)


class TimelineDashboardView:
    """Keep large patient tables out of Plotly and render only the selected view."""

    def __init__(self, patients: list[dict], cohort: dict, related_plots: list[dict],
                 theme: str, layout: DashboardLayout, compare_all: bool) -> None:
        self.patients = patients
        self.cohort = cohort
        self.related_plots = related_plots
        self.theme = theme
        self.layout = layout
        self.compare_all = compare_all

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")

    def head(self) -> str:
        dark = self.theme == "dark"
        background, surface, ink, muted, border = (
            ("#0b0d12", "#151820", "#f2f4f8", "#a6adbd", "#303642") if dark else
            ("#f5f7fb", "#fff", "#1b2432", "#536176", "#d9e0ea")
        )
        return f"""<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--timeline-bg:{background};--timeline-surface:{surface};--timeline-ink:{ink};--timeline-muted:{muted};--timeline-border:{border};--timeline-panel-height:{self.layout.panel_max_height}px}}
body{{background:var(--timeline-bg);color:var(--timeline-ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}}
.timeline-shell{{max-width:1780px;margin:0 auto;padding:16px clamp(12px,3vw,40px) 32px}}
.timeline-header{{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;position:sticky;top:0;z-index:10;background:var(--timeline-bg);padding:8px 0 12px}}
.timeline-header h1{{font-size:22px;letter-spacing:-.03em;margin:0}}
.timeline-header p,.timeline-hint{{color:var(--timeline-muted);margin:4px 0 0}}
.timeline-selector{{display:grid;gap:4px;font-size:12px;font-weight:600;color:var(--timeline-muted)}}
.timeline-selector select{{min-width:220px;max-width:300px;padding:10px 32px 10px 12px;border:1px solid var(--timeline-border);border-radius:8px;background:var(--timeline-surface);color:var(--timeline-ink);font:inherit;font-size:14px}}
.timeline-selector select:focus-visible,.timeline-series-menu :focus-visible,.timeline-events button:focus-visible{{outline:2px solid #7170ff;outline-offset:2px}}
.timeline-controls{{display:flex;align-items:end;gap:10px;flex-wrap:wrap}}
.timeline-series-menu{{position:relative;min-width:210px}}
.timeline-series-menu>summary{{list-style:none;cursor:pointer;padding:10px 32px 10px 12px;border:1px solid var(--timeline-border);border-radius:8px;background:var(--timeline-surface);font-weight:600}}
.timeline-series-menu>summary::after{{content:'⌄';float:right;margin-left:12px}}
.timeline-series-menu[open]>summary{{border-color:#7170ff}}
.timeline-series-popover{{position:absolute;right:0;top:calc(100% + 4px);z-index:30;width:min(360px,90vw);padding:12px;border:1px solid var(--timeline-border);border-radius:10px;background:var(--timeline-surface);box-shadow:0 14px 34px #0003}}
.timeline-series-popover input[type="search"]{{box-sizing:border-box;width:100%;padding:8px;border:1px solid var(--timeline-border);border-radius:6px;background:var(--timeline-bg);color:var(--timeline-ink)}}
.timeline-series-popover label{{display:flex;align-items:flex-start;gap:8px;padding:7px 2px;overflow-wrap:anywhere;cursor:pointer}}
.timeline-series-options{{max-height:240px;overflow-y:auto;border-top:1px solid var(--timeline-border)}}
.timeline-series-options label[hidden],.timeline-more[hidden],#timeline-series-empty[hidden]{{display:none}}
.timeline-series-popover p{{margin:4px 0 0}}
.timeline-milestones{{display:flex;gap:0;list-style:none;margin:12px 0 0;padding:0;overflow-x:auto}}
.timeline-milestones li{{position:relative;min-width:180px;flex:1;padding:14px 12px 8px 20px;border-top:2px solid var(--timeline-border)}}
.timeline-milestones li::before{{content:'';position:absolute;top:-6px;left:4px;width:10px;height:10px;border-radius:50%;background:#7170ff}}
.timeline-milestones strong,.timeline-milestones span{{display:block}}
.timeline-milestones span{{color:var(--timeline-muted);font-size:12px;margin-top:3px}}
.timeline-events{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin-top:16px}}
.timeline-phase{{min-width:0;border:1px solid var(--timeline-border);border-radius:8px;padding:10px}}
.timeline-phase h3{{margin:0 0 6px;font-size:13px}}
.timeline-phase ol{{list-style:none;padding:0;margin:0;max-height:var(--timeline-panel-height);overflow-y:auto}}
.timeline-phase li{{border-left:2px solid #7170ff;padding:3px 0 3px 10px;margin:8px 0}}
.timeline-phase li[data-kind="boundary"]{{border-color:#e1a84e;font-weight:600}}
.timeline-phase button{{display:block;width:100%;text-align:left;border:0;background:none;color:var(--timeline-ink);font:inherit;cursor:pointer;padding:2px 0;overflow-wrap:anywhere}}
.timeline-phase button:hover{{color:#7170ff}}
.timeline-phase time{{font-size:12px;color:var(--timeline-muted)}}
.timeline-more{{margin-top:8px;padding:7px 10px;border:1px solid var(--timeline-border);border-radius:6px;background:var(--timeline-surface);color:var(--timeline-ink);cursor:pointer}}
.timeline-demographics{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
.timeline-demographics h2{{width:100%}}
.timeline-demographics .timeline-stat{{min-width:140px;min-height:0}}
.timeline-card{{background:var(--timeline-surface);border:1px solid var(--timeline-border);border-radius:10px;padding:14px;margin-top:12px}}
.timeline-chart{{padding:0 8px;overflow:hidden}}
.timeline-card h2,.timeline-card summary{{font-size:14px;font-weight:650;margin:0;cursor:pointer}}
.timeline-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.timeline-stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin:12px 0}}
.timeline-stat{{border:1px solid var(--timeline-border);border-radius:8px;padding:10px;min-height:54px}}
.timeline-stat span{{display:block;font-size:11px;color:var(--timeline-muted)}}
.timeline-stat strong{{font-size:15px;font-weight:600;overflow-wrap:anywhere}}
.table-scroll{{max-height:var(--timeline-panel-height);overflow:auto;margin-top:10px}}
.table-scroll table{{width:100%;border-collapse:collapse;text-align:left}}
.table-scroll th,.table-scroll td{{padding:7px 10px;border-bottom:1px solid var(--timeline-border);vertical-align:top;overflow-wrap:anywhere}}
.table-scroll th{{background:var(--timeline-surface);position:sticky;top:0;z-index:1}}
.timeline-card[hidden]{{display:none}}
.timeline-card iframe{{width:100%;height:450px;border:0;margin-top:10px;background:var(--timeline-bg)}}
.timeline-card details details{{margin:8px 0;border-top:1px solid var(--timeline-border);padding-top:8px}}
@media(max-width:850px){{.timeline-events{{grid-template-columns:1fr}}.timeline-header{{position:static}}}}
@media(max-width:650px){{.timeline-shell{{padding:12px}}.timeline-header h1{{font-size:18px}}.timeline-chart{{overflow-x:auto}}}}
</style>"""

    def before_plot(self) -> str:
        return """<main class="timeline-shell">
<header class="timeline-header"><div><h1>Patient timeline</h1><p>Operation-relative hours · pseudonymous patients · no calendar dates</p></div>
<div class="timeline-controls"><label class="timeline-selector" for="timeline-patient">Patient
<select id="timeline-patient" aria-label="Select patient to display"></select></label>
<details id="timeline-series-menu" class="timeline-series-menu"><summary id="timeline-series-count">Time series</summary>
<div class="timeline-series-popover"><input id="timeline-series-search" type="search" placeholder="Find a time series" aria-label="Find a time series">
<label><input id="timeline-series-all" type="checkbox" checked>Select all time series</label>
<div id="timeline-series-options" class="timeline-series-options" role="group" aria-label="Plotted time series"></div>
<p class="timeline-hint">Selections apply to the plot, not the event timeline.</p></div></details></div></header>
<section class="timeline-card timeline-chart" aria-label="During-operative time-series chart">
<p id="timeline-series-empty" class="timeline-hint" hidden>No time series selected or available for this patient.</p>"""

    def after_plot(self) -> str:
        return """</section>
<section class="timeline-card" aria-label="Main events timeline"><h2>Main events · relative to operation start</h2>
<p class="timeline-hint">Operation boundaries and recorded events across all three phases. Select an event for details.</p>
<ol id="timeline-milestones" class="timeline-milestones"></ol>
<div id="timeline-duration-stats" class="timeline-stats" aria-label="Configured stay durations"></div>
<p class="timeline-hint">Confirm event meaning before interpreting stay durations as hospital stay.</p>
<div id="timeline-events" class="timeline-events" aria-label="Pre-, during-, and post-operative events"></div>
<section id="timeline-event-details" class="timeline-card" hidden aria-live="polite"><h2>Selected timeline event</h2>
<p data-role="event" class="timeline-hint">Select a pre-, during-, or post-operative event.</p>
<p data-role="time" class="timeline-hint">Time is relative to operation start.</p>
<div data-role="details" class="table-scroll"></div></section></section>
<section id="timeline-demographics" class="timeline-card timeline-demographics" aria-label="Untimed demographics">
<h2>Demographics <span class="timeline-hint">· No timestamp; not positioned on the timeline</span></h2></section>
<div id="timeline-stats" class="timeline-stats" aria-label="Patient overview"></div>
<div class="timeline-grid">
<details id="timeline-pre" class="timeline-card"><summary>Pre-operative data and events</summary><div class="table-scroll"></div></details>
<details id="timeline-post" class="timeline-card"><summary>Post-operative data and events</summary><div class="table-scroll"></div></details>
</div>
<details id="timeline-summary" class="timeline-card"><summary>Full patient and operation summary</summary><div class="table-scroll"></div></details>
<details id="study-plots" class="timeline-card"><summary>Study plots</summary><div data-role="plots"></div></details>
</main>"""

    def script(self) -> str:
        payload = self._json({
            "patients": [{key: patient[key] for key in (
                "patient_id", "pre_rows", "post_rows", "summary_rows", "demographics", "milestones", "timeline_events", "duration_stats"
            )} for patient in self.patients],
            "cohort": self.cohort,
            "plots": self.related_plots,
            "compare": self.compare_all,
        })
        return """
(() => {
const ready = () => {
  const plot = document.getElementById('{plot_id}');
  if (!plot) return;
  const data = __PAYLOAD__;
  const byId = new Map(data.patients.map(patient => [patient.patient_id, patient]));
  const selector = document.getElementById('timeline-patient');
  for (const id of [...data.patients.map(patient => patient.patient_id), ...(data.compare ? ['compare'] : [])]) {
    const option = document.createElement('option');
    option.value = id;
    option.textContent = id === 'compare' ? 'Compare all patients' : id;
    selector.append(option);
  }
  selector.value = data.compare ? 'compare' : data.patients[0].patient_id;
  const menu = document.getElementById('timeline-series-menu');
  const seriesAll = document.getElementById('timeline-series-all');
  const seriesOptions = document.getElementById('timeline-series-options');
  const seriesSearch = document.getElementById('timeline-series-search');
  const seriesCount = document.getElementById('timeline-series-count');
  const seriesEmpty = document.getElementById('timeline-series-empty');
  const seriesNames = [...new Set(plot.data.map(trace => trace.meta.series))].sort((a, b) => a.localeCompare(b));
  const selectedSeries = new Set(seriesNames);
  function updateSeries() {
    const id = selector.value;
    const visible = plot.data.map(trace =>
      (id === 'compare' || trace.meta.patient === id) && selectedSeries.has(trace.meta.series));
    if (visible.length) Plotly.restyle(plot, 'visible', visible);
    seriesEmpty.hidden = visible.some(Boolean);
    seriesCount.textContent = `Time series · ${selectedSeries.size}/${seriesNames.length}`;
    seriesAll.checked = selectedSeries.size === seriesNames.length && seriesNames.length > 0;
    seriesAll.indeterminate = selectedSeries.size > 0 && selectedSeries.size < seriesNames.length;
  }
  seriesNames.forEach(name => {
    const label = document.createElement('label');
    const input = document.createElement('input'); input.type = 'checkbox'; input.checked = true;
    input.value = name;
    const title = document.createElement('span'); title.textContent = name;
    label.append(input, title); seriesOptions.append(label);
    input.addEventListener('change', () => {
      if (input.checked) selectedSeries.add(name); else selectedSeries.delete(name);
      updateSeries();
    });
  });
  seriesAll.addEventListener('change', () => {
    selectedSeries.clear();
    if (seriesAll.checked) seriesNames.forEach(name => selectedSeries.add(name));
    seriesOptions.querySelectorAll('input').forEach(input => { input.checked = selectedSeries.has(input.value); });
    updateSeries();
  });
  seriesSearch.addEventListener('input', () => {
    const query = seriesSearch.value.trim().toLocaleLowerCase();
    seriesOptions.querySelectorAll('label').forEach(label => {
      label.hidden = !label.textContent.toLocaleLowerCase().includes(query);
    });
  });
  if (!seriesNames.length) { seriesAll.disabled = true; seriesSearch.disabled = true; menu.open = false; }
  updateSeries();
  const milestoneList = document.getElementById('timeline-milestones');
  const durationStats = document.getElementById('timeline-duration-stats');
  const eventLanes = document.getElementById('timeline-events');
  const demographics = document.getElementById('timeline-demographics');
  const sections = [
    {node: document.getElementById('timeline-pre'), key: 'pre', rows: 'pre_rows', headers: ['Data or event','Value / summary','Time']},
    {node: document.getElementById('timeline-post'), key: 'post', rows: 'post_rows', headers: ['Data or event','Value / summary','Time']},
    {node: document.getElementById('timeline-summary'), key: 'summary', rows: 'summary_rows', headers: ['Data or event','Value / summary','Time']},
  ];
  let selected = data.compare ? null : data.patients[0].patient_id;
  const stats = document.getElementById('timeline-stats');
  const panel = document.getElementById('timeline-event-details');
  const details = panel.querySelector('[data-role="details"]');
  const eventLabel = panel.querySelector('[data-role="event"]');
  const timeLabel = panel.querySelector('[data-role="time"]');
  function table(target, headers, rows) {
    target.replaceChildren();
    const element = document.createElement('table');
    const heading = document.createElement('thead');
    const tr = heading.insertRow();
    headers.forEach(label => { const th = document.createElement('th'); th.textContent = label; tr.append(th); });
    element.append(heading);
    const body = document.createElement('tbody');
    rows.forEach(values => { const row = body.insertRow(); values.forEach(value => {
      const cell = row.insertCell(); cell.textContent = String(value ?? '—');
    }); });
    element.append(body);
    target.append(element);
  }
  function resetEvent() {
    panel.hidden = true;
    eventLabel.textContent = 'Select a pre-, during-, or post-operative event.';
    timeLabel.textContent = 'Time is relative to operation start.';
    details.replaceChildren();
  }
  function showEvent(item, patient) {
    panel.hidden = false;
    eventLabel.textContent = `${patient.patient_id}: ${item.phase}: ${item.value || item.name}`;
    timeLabel.textContent = item.hours === null ? 'Operation start unavailable' :
      `${item.hours > 0 ? '+' : ''}${item.hours.toFixed(2)} h from operation start (no calendar date)`;
    if (item.details && item.details.length) {
      table(details, ['Data','Value','Unit'], item.details.map(value =>
        [value.name || value.parameter_id || '', value.value ?? '', value.unit || '—']));
    } else {
      details.replaceChildren();
      const empty = document.createElement('p'); empty.className = 'timeline-hint';
      empty.textContent = 'No mapped measurements recorded with this event.';
      details.append(empty);
    }
    panel.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  }
  function renderEvents(patient) {
    eventLanes.replaceChildren();
    const phases = ['Pre-operative', 'During-operative', 'Post-operative'];
    const postEnd = patient.milestones.find(item => item.name === 'Post-operative section end')?.hours;
    const laterPhase = 'After post-operative end';
    if (postEnd !== null && postEnd !== undefined &&
        patient.timeline_events.some(item => item.hours > postEnd)) phases.push(laterPhase);
    const boundaries = {
      'Operation start': 'During-operative',
      'Operation end': 'Post-operative',
      'Post-operative section end': 'Post-operative'
    };
    phases.forEach(phase => {
      const events = patient.timeline_events.filter(item =>
        phase === laterPhase ? item.hours > postEnd :
          item.phase === phase && (phase !== 'Post-operative' || postEnd == null || item.hours <= postEnd))
        .map(item => ({...item, phase, kind: 'event'}));
      const markers = patient.milestones.filter(item => boundaries[item.name] === phase).map(item =>
        ({...item, phase, kind: 'boundary'}));
      const entries = [...events, ...markers].sort((a, b) =>
        (a.hours ?? Infinity) - (b.hours ?? Infinity) || (a.kind === 'boundary' ? -1 : 1));
      const lane = document.createElement('section'); lane.className = 'timeline-phase';
      const heading = document.createElement('h3');
      heading.textContent = `${phase} · ${events.length} ${events.length === 1 ? 'event' : 'events'}`;
      const list = document.createElement('ol'); list.setAttribute('aria-label', `${phase} main events`);
      lane.append(heading, list);
      const more = document.createElement('button'); more.type = 'button'; more.className = 'timeline-more';
      let shown = 0;
      const appendBatch = () => {
        entries.slice(shown, shown + 50).forEach(item => {
          const row = document.createElement('li'); row.dataset.kind = item.kind;
          const button = document.createElement('button'); button.type = 'button';
          button.textContent = item.value || item.name;
          button.addEventListener('click', () => showEvent(item, patient));
          const time = document.createElement('time');
          time.textContent = item.hours === null ? 'Unavailable' : `${item.hours > 0 ? '+' : ''}${item.hours.toFixed(2)} h`;
          row.append(button, time); list.append(row);
        });
        shown = Math.min(shown + 50, entries.length);
        more.hidden = shown >= entries.length;
        more.textContent = `Show more events (${entries.length - shown} remaining)`;
      };
      more.addEventListener('click', appendBatch);
      appendBatch();
      if (!entries.length) {
        const empty = document.createElement('p'); empty.className = 'timeline-hint';
        empty.textContent = 'No timed events in this phase.'; lane.append(empty);
      }
      lane.append(more); eventLanes.append(lane);
    });
  }
  function renderContext(patient) {
    milestoneList.replaceChildren();
    durationStats.replaceChildren();
    eventLanes.replaceChildren();
    demographics.querySelectorAll('.timeline-stat, .timeline-empty').forEach(element => element.remove());
    if (!patient) {
      const prompt = document.createElement('li');
      prompt.textContent = 'Choose a patient to see their operation milestones.';
      milestoneList.append(prompt);
      const empty = document.createElement('p');
      empty.className = 'timeline-empty timeline-hint';
      empty.textContent = 'Choose a patient to see untimed demographics.';
      demographics.append(empty);
      return;
    }
    patient.milestones.forEach(item => {
      const step = document.createElement('li');
      const name = document.createElement('strong'); name.textContent = item.name;
      const time = document.createElement('span');
      time.textContent = item.hours === null ? 'Unavailable' :
        `${item.hours > 0 ? '+' : ''}${item.hours.toFixed(2)} h`;
      step.append(name, time); milestoneList.append(step);
    });
    patient.duration_stats.forEach(([label, value]) => {
      const card = document.createElement('div'); card.className = 'timeline-stat';
      const caption = document.createElement('span'); caption.textContent = label;
      const display = document.createElement('strong'); display.textContent = value;
      card.append(caption, display); durationStats.append(card);
    });
    renderEvents(patient);
    const available = patient.demographics.filter(([, value]) => value !== 'Unavailable');
    available.forEach(([name, value]) => {
      const card = document.createElement('div'); card.className = 'timeline-stat';
      const label = document.createElement('span'); label.textContent = name;
      const text = document.createElement('strong'); text.textContent = value;
      card.append(label, text); demographics.append(card);
    });
    if (available.length !== patient.demographics.length) {
      const empty = document.createElement('p'); empty.className = 'timeline-empty timeline-hint';
      empty.textContent = `${patient.demographics.length - available.length} unavailable demographic fields`;
      demographics.append(empty);
    }
    if (!patient.demographics.length) {
      const empty = document.createElement('p'); empty.className = 'timeline-empty timeline-hint';
      empty.textContent = 'No untimed classes are configured.'; demographics.append(empty);
    }
  }
  function renderTables() {
    sections.forEach(({node, key, rows, headers}) => {
      const target = node.querySelector('.table-scroll');
      target.replaceChildren();
      if (!node.open) return;
      if (selected) table(target, headers, byId.get(selected)[rows]);
      else table(target, data.cohort[key].headers, data.cohort[key].rows);
    });
  }
  sections.forEach(({node}) => node.addEventListener('toggle', renderTables));
  function renderPatient(id) {
    const patient = byId.get(id);
    if (!patient) return;
    selected = id;
    stats.replaceChildren();
    const values = new Map(patient.summary_rows.map(row => [row[0], row[1]]));
    ['Operation duration', 'Death status', 'Distance from operation to death'].forEach(label => {
      const card = document.createElement('div'); card.className = 'timeline-stat';
      const caption = document.createElement('span'); caption.textContent = label;
      const value = document.createElement('strong'); value.textContent = values.get(label) || 'Unavailable';
      card.append(caption, value); stats.append(card);
    });
    renderContext(patient);
    renderTables();
    resetEvent();
  }
  function renderCohort() {
    selected = null;
    stats.replaceChildren();
    const card = document.createElement('div'); card.className = 'timeline-stat';
    const caption = document.createElement('span'); caption.textContent = 'Patients compared';
    const value = document.createElement('strong'); value.textContent = String(data.patients.length);
    card.append(caption, value); stats.append(card);
    renderContext(null);
    renderTables();
    resetEvent();
  }
  if (data.compare) renderCohort(); else renderPatient(data.patients[0].patient_id);
  selector.addEventListener('change', () => {
    const id = selector.value;
    if (id !== 'compare' && !byId.has(id)) return;
    updateSeries();
    Plotly.relayout(plot, {title: `Perioperative dashboard — ${id === 'compare' ? 'all patients' : id}`});
    if (id === 'compare') renderCohort(); else renderPatient(id);
  });
  const plots = document.querySelector('#study-plots [data-role="plots"]');
  if (!data.plots.length) { const message = document.createElement('p');
    message.textContent = 'No related study plots were generated.'; plots.append(message); }
  data.plots.forEach(item => {
    const section = document.createElement('details');
    const label = document.createElement('summary'); label.textContent = item.label;
    const frame = document.createElement('iframe'); frame.dataset.src = item.src;
    frame.title = item.label; frame.loading = 'lazy';
    section.addEventListener('toggle', () => { if (section.open && !frame.src) frame.src = frame.dataset.src; });
    section.append(label, frame); plots.append(section);
  });
};
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', ready, {once: true});
} else {
  ready();
}
})();
""".replace("__PAYLOAD__", payload)
