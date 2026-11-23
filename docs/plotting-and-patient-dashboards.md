# Plotting and individual patient dashboards

The plotting CLI reads phase-separated extraction CSVs under `results/parameters/`. It also reads `summary.csv` for operation boundaries and the extracted `epic_event` observations for timeline markers. Patient HTML dashboards remain separate from cohort-level study plots.

## Heatmap-specific parameter selection

A heatmap **does not need to include all parameters in `parameters.ini`**. Define the subset in `config/plotting.ini`:

```ini
[value_heatmaps]
enabled = true
aggregation = patient_mean
parameter_ids = age, bmi, kreatinine
```

`parameter_ids = all` selects all eligible numeric and timeseries parameters. The heatmap selection is independent of `[selection] parameter_ids` used by other plots and dashboards. The `[selection] phases` and patient filter still apply. Unknown heatmap parameter IDs raise a configuration error. The heatmap excludes categorical (`str`, `bool`) parameters, and normalizes each numeric parameter column independently for visualization; hover retains the original value. A configured ID with no numeric values is displayed as missing, not silently filled. Heatmap scaling is descriptive, not a training transformation.

To plot only heatmaps, set `enabled=false` in `[distributions]`, `[frequencies]`, `[missingness]`, `[time_series]`, `[patient_timelines]` and `[patient_dashboards]`.

## Individual patient HTML dashboards

```ini
[patient_dashboards]
enabled = true
output_directory = results/plots/patients
include_plotlyjs = true
```

The main timeline displays measurements against minutes relative to the configured operation start, an operation-start line at `0 min`, an operation-end line at the patient-specific `operation_end_minutes` from `summary.csv`, and Epic event markers. If the end boundary is absent, it is shown as unavailable rather than inferred. Event tables, patient attributes and phase tables are included. The output is one HTML per patient; no cohort study plots are embedded in these files.

## Running

```powershell
uv run python -m src.cli parameters-extraction --input data/ --config config/parameters.ini
uv run python -m src.cli plotting --config config/plotting.ini
```

Plotting requires a completed extraction run, and changing `parameters.ini` requires rerunning extraction. A derivation such as `operation_duration` is generated during extraction; heatmap selection controls only visualization. For a duration heatmap, include `operation_duration` and the `intraoperative` phase.

## Troubleshooting

- **Heatmap missing BMI:** Check that BMI was extracted or derived; inspect `value_origin`, weight/height values and units.
- **Heatmap missing operation duration:** Check both operation boundaries and the intraoperative phase filter.
- **End-of-operation line missing:** Check `results/parameters/summary.csv` for `operation_end_minutes` for that patient.
- **No Epic events:** Check the extraction output for `parameter_id=epic_event` and valid event timestamps.
- **Unexpected extra heatmap columns:** Set `[value_heatmaps] parameter_ids` explicitly rather than `all`.
- **Patient values differ across plots:** Inspect the chosen aggregation (`patient_mean`, `patient_median`, `patient_min`, `patient_max`) and the selected clinical phases.


## Independent missing-data heatmap selection

The missing-data heatmap is controlled by `[missingness] parameter_ids` and
**does not inherit** `[selection] parameter_ids` or `[value_heatmaps] parameter_ids`.
The general selection still controls distributions, frequencies, time-series
and timelines. The value heatmap has its own numeric-only selection.

```ini
[selection]
phases = preoperative, intraoperative, postoperative
parameter_ids = kreatinine

[missingness]
enabled = true
parameter_ids = age, sex, weight, height, bmi

[value_heatmaps]
enabled = true
aggregation = patient_mean
parameter_ids = bmi, kreatinine
```

Use `parameter_ids = all` alone for all configured parameters within the
selected phases. IDs apply to every selected phase of a multiphase parameter.
Unknown IDs, an empty list, or mixing `all` with named IDs cause an error.
Unlike numeric value heatmaps, the missing-data heatmap accepts categorical
parameters such as `sex`.

A cell is **present** if at least one valid extracted observation exists for
that patient and parameter/phase; otherwise it is **missing**. The plot does
not distinguish a missing raw column from an invalid reading or an
unavailable operation boundary. Derived values count as present when
successfully calculated; inspect `value_origin` to distinguish them from
recorded values.

The resulting file is `results/plots/heatmaps/missingness.html` unless the
plot output directory is configured differently.

## Independent phase selection for heatmaps and availability

The three plots below have **independent** parameter and phase selection.
They do not inherit `[selection] phases` or `[selection] parameter_ids`.

```ini
[missingness]
enabled = true
parameter_ids = age, sex, bmi
phases = preoperative

[parameter_availability]
enabled = true
parameter_ids = kreatinine, bmi
phases = preoperative, postoperative

[value_heatmaps]
enabled = true
parameter_ids = bmi, kreatinine
phases = all
aggregation = patient_mean
```

The `phases` option can be omitted or set to `all` to include **all
three clinical phases** (preoperative, intraoperative, postoperative).
Alternatively, provide a comma-separated list of valid phases. Mixing
`all` with named phases, unknown names, and duplicate phases is invalid.
A parameter configured for multiple phases appears once per selected
phase. If no parameter-phase pairs are eligible, the command reports an error.

`parameter_ids = all` selects all configured parameter IDs that have
eligible phases. Unknown IDs and combining `all` with other IDs are errors.

The missing-data heatmap (`heatmaps/missingness.html`) and the
parameter-availability plot (`heatmaps/availability.html`) both accept
numeric and categorical parameters. They share the same presence
criterion: at least one valid extracted observation for that
patient/parameter/phase. They can use different selections. The value
heatmap (`heatmaps/values.html`) accepts numeric/timeseries parameters
only and min-max scales per parameter-phase column for visualization.

These plots reflect **extracted** values. Missing operation boundaries
can prevent extraction of timed observations; absence in a heatmap
does not prove a value was never recorded in the source data.
