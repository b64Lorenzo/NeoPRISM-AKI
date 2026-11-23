# Phase-aware parameter extraction

## Aim

Each input patient family may contain one or more CSV parts, such as `ref_001.csv`, `ref_001_1.csv` and `ref_001_2.csv`. The extractor groups these into one pseudonymous patient ID and exports **three phase-specific observation CSVs**:

```text
results/parameters/
  preoperative/<patient_id>.csv
  intraoperative/<patient_id>.csv
  postoperative/<patient_id>.csv
  summary.csv
  manifest.csv
  patient_events_summary.csv
```

## Temporal definition

For an observation acquired at source timestamp `t`, the exported relative time is

\[\tau=\frac{t-t_{\mathrm{start}}}{60}\;\mathrm{minutes}.\]

Here `t` and `t_start` are internally represented as seconds; the exported `time_minutes` is negative before operation start, zero at start and positive afterward. The extractor does not export calendar timestamps.

When both boundaries are available, the phase membership is

- **Preoperative:** `t < operation_start`
- **Intraoperative:** `operation_start <= t < operation_end`
- **Postoperative:** `operation_end <= t < postoperative_end`, or no upper bound if that boundary is not configured.

**Missing required boundaries:** time-assigned observations cannot be reliably assigned to all phases. The extracted event frequency inventory may still include events; the dashboard does not fabricate an operation end.

## Operation boundaries

The uploaded extractor searches configured `fields` for case-folded **substring** matches against each comma-separated phrase. Compound Epic event cells may include `|` separators. A single matching phrase can therefore identify a compound string.

```ini
[boundary.operation_start]
fields = epic.EVENTS.EVENT
phrases = Aankomst OK | Start anesthesie, surgery start, operation start

[boundary.operation_end]
fields = epic.EVENTS.EVENT
phrases = Afronden zorg | Stop anesthesie, surgery end, operation end
```

The first qualifying start and first end **after** that start are used. In your Epic inventory, two different departure strings exist (`Vertrek naar IC (of_Afdeling)` and `Vertrek naar IC of Afdeling`), so matching a stable component such as `Afronden zorg | Stop anesthesie` can capture both. However, never assume anesthesia end is the actual time of surgical closure without validating the clinical event semantics. Avoid overly broad phrases that might falsely match unrelated events.

## Parameter INI schema

```ini
[parameter.age]
phase = preoperative
source_fields = epic.DEMOGRAPHICS.age
type = numeric
unit = months
timed = false

[parameter.kreatinine]
phase = preoperative, postoperative
source_fields = epic.LAB.Kreatinine (umol/L)
type = timeseries
unit = umol/L
timed = true
```

- `phase`: one or more of `preoperative`, `intraoperative`, `postoperative`; multi-phase entries must be timed.
- `type`: `numeric`, `timeseries`, `str` or `bool`.
- `source_fields`: comma-separated aliases, matched against source CSV headers.
- `timed=false`: only preoperative; the value is shown as untimed and is not located on the chart's time axis.
- `unit`: a display string; the extractor **does not automatically convert recorded measurement units**, except in a configured derived formula such as BMI.

## Output schema

The observation CSV header is:

```csv
time_minutes,parameter_id,value,unit,type,source_field,value_origin
```

`value_origin` is `observed` for raw data and e.g. `derived_bmi_from_weight_height` for a BMI calculation. An empty `time_minutes` means an untimed clinical attribute, **not** operation time zero.

The `summary.csv` includes `patient_id`, `source_files`, `operation_start_minutes`, `operation_end_minutes`, `postoperative_end_minutes`, and the number of rows in each phase. The dashboard draws its vertical operation-end marker from the `operation_end_minutes` field in this summary.

## Epic event visibility

The extractor automatically writes timestamped `epic.EVENTS.EVENT` values as `parameter_id=epic_event` into the corresponding phase CSV, with `value_origin=observed`. The separate `patient_events_summary.csv` is **counts only**; it does not encode timing and cannot recreate chronological events. Plotting loads the timed `epic_event` rows even if they are not included in the numeric parameter selection.

If you change the source CSV or boundary phrases, rerun `parameters-extraction`; simply replotting cannot recover events or boundaries that were never exported.

## Data integrity

The pipeline assumes `unixtime` is Unix seconds or milliseconds; ISO timestamp fields are also accepted. Confirm timezone and source timestamp interpretation before clinical analysis. For outcome modelling, ensure no postoperative variable enters a preoperative prediction feature. This module does not assign KDIGO stages or provide outcome labels by itself.

## Derived parameter registry

`derive = bmi` and `derive = operation_duration` are available. They are calculated after raw patient observations have been extracted, without altering original measurements. The dispatcher in `src/preprocessing/parameters/derivation.py` is extensible with `@register("method")`. Operation duration is an intraoperative summary with blank `time_minutes`, derived from the configured operation boundaries. See [Derived parameters](derived-parameters.md) for details and precautions.
