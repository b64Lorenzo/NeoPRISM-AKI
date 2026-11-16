# Patient-vector workflow and artifacts

The CLI creates one feature vector and one outcome per patient filename family. Pre-operative features become model inputs; postoperative evidence supplies labels and is kept separate. The perceptron joins the files by `patient_id`, never by row order. See the [training guide](perceptron-training.md) for fitting and validation.

## Run

```powershell
py -3 -m src.cli analyze "C:\path\to\patient-csvs" --output "C:\path\to\results"
py -3 -m src.cli distributions "C:\path\to\patient-csvs" --output "C:\path\to\results" --phase all --bins 20
```

Directories are searched recursively. Use a folder containing only the intended cohort. `--outcome-column` selects or prioritizes a direct outcome column; `--success-values` and `--failure-values` override its exact category lists. Confirm the cohort coding before overriding the XML rules.

## Patient families and privacy

Files are grouped only within the same directory. A short final numeric suffix is treated as a part: `ref_0002.csv`, `ref_0002_1.csv`, and `ref_0002_2.csv` form one family; `case_1.csv` and `case_2.csv` also group when `case.csv` is absent. A final numeric suffix of three or more digits, such as `_0002`, remains part of the base name; shorter suffixes are treated as continuations. Identical names in different directories are not merged.

Each file uses its own header and delimiter; columns align by header name. Files are processed in sorted order, and overlapping or duplicate rows are **not** deduplicated. The workflow assumes one index operation per family; separate operations need an episode key before interpretation. Pseudonymous IDs are assigned by the sorted family list and are not stable across runs. The source-to-ID mapping is not saved; original filenames are omitted from patient-level outputs and logs.

## Features, boundaries, and labels

- `patient_vectors.csv` has `patient_id` plus configured pre-operative features only—no label. The default model configuration omits `operation_diagnosis_proxy`.
- The earliest configured pre-operative `endEvent` marks operation start; the first later during-operative `endEvent` marks operation end. Pre-operative readings are before start, during readings are in `[start, end)`, and timestamp-scoped postoperative evidence starts at end. A missing marker does not shift evidence into another phase. Timestamps accept ISO values or numeric `unixtime` (seconds, with magnitude-detected milliseconds). See [editable operation boundaries](xml-pipeline-configuration.md#editable-operation-boundaries).
- Time-varying features use the last valid reading before the cutoff. Without a start marker, they remain missing; untimed/static metadata is used only when the family has exactly one distinct non-missing value. Numeric features keep the numeric component before the first `|`; categorical values are retained.
- Missing values are not imputed except BMI, which may be derived from pre-operative weight and height. The XML defaults assume kg/cm; verify units with the data dictionary. The perceptron fits numeric scaling and categorical encodings on training rows only; rows missing selected inputs are excluded.
- `patient_outcomes.csv` stores a target (`-1` failure, `+1` success, or `NA` unclassified), status, and label basis. Failure evidence wins; `NA` is excluded, not a third class. Criteria and their provisional/pending status are listed in [post-operative outcomes](postoperative-outcomes.md).

Postoperative measurement aggregation and optional time windows are configured on each parameter. Measurements are review/label evidence, not predictors. By default, timestamped evidence starts at operation end; `record_scope=patient` permits patient-level metadata from any row. See the [XML guide](xml-pipeline-configuration.md#phase-owned-definitions-and-selection).

## Optional analysis and plots

`exportTimeSeries=true` writes `time_series.csv` with `patient_id,phase,parameter_id,t_seconds,value`. It includes selected numeric during/postoperative readings only, uses operation start/end as the respective zero points, and contains no absolute timestamps. No zero-time reading is invented; missing timing or nonnumeric values are skipped. This export does not change model inputs or labels. Plotting reads the generated CSV and displays elapsed hours; see [time-series export and plotting](xml-pipeline-configuration.md#phase-relative-time-series-export).

The optional patient timeline shows operation-relative hours, keeps untimed demographics off the timeline, and omits absolute dates. Its searchable series selector supports individual or all traces; events after the configured stay endpoint appear in a follow-up lane. Numeric traces are independently scaled for display, with original values on hover. Summary panels include operation boundaries/duration, counts, timing gaps, configured ICU and hospital durations, and death status/recorded days when available. Study-plot embeds and patient comparisons are configurable. Source event vocabulary, ICU date fields, and timezone assumptions require confirmation. The HTML contains patient-level values; keep it and its local Plotly bundle in approved storage.

Feature heatmaps use a deterministic sample of up to 500 patients. Numeric columns are min–max scaled independently (`0` to `1`; constant columns to `0.5`); categorical values are one-hot encoded for display, and missing/unresolved cells are gray. These transformations do not change CSVs. The patient-missingness map uses `1` for missing and `0` for present; it defaults to pre-operative features, but XML can independently select phase, parameter IDs, and mapped classes. A recorded zero is present. Plot filters change only the visualization, not extracted features or model inputs; see [plot settings](xml-pipeline-configuration.md#distribution-and-missingness-plot-selection).

## Output files

| File | Contents |
|---|---|
| `patient_vectors.csv` | Patient ID and pre-operative features; no target. |
| `patient_outcomes.csv` | Target, status, and label basis from postoperative evidence. |
| `duringoperative_measurements.csv` | Optional; latest mapped value in `[start, end)`, never a model input. |
| `postoperative_measurements.csv` | Aggregated postoperative evidence; never a model input. |
| `time_series.csv` | Optional phase-relative numeric readings; no dates; never a model input. |
| `training_set.csv` | Training features and true target. |
| `validation_set.csv` | Validation features and prediction; no true target. |
| HTML and report files | `preoperative_heatmap.html`, `preoperative_missingness_heatmap.html` (default, output configurable), `postoperative_heatmap.html`, `patient_timeline_dashboard.html`, and `extraction_report.md`; distributions and time-series plots are optional. |

Existing files in the output directory are not removed between runs; use a fresh directory when you need a clean result. Tests use synthetic fixtures only and do not establish clinical validity. Keep cohort CSVs, reports, plots, and logs in approved storage and out of version control.