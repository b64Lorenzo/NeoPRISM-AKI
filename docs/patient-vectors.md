# Patient-vector workflow and artifacts

The CLI treats **same-directory files in a numbered filename family as one patient time series**. For example, `ref_0002.csv`, `ref_0002_1.csv`, and `ref_0002_2.csv` produce one feature vector and one outcome. It extracts configured pre-operative values and stores postoperative-derived targets separately. The perceptron uses selected pre-operative inputs; parameter IDs listed in the model's XML `trainingValidation/omit` section are excluded. The default configuration omits `operation_diagnosis_proxy`. Postoperative evidence is reserved for label creation and validation. Targets are `-1` for failure, `+1` for success, and `NA` only when evidence is missing or no criterion matches. See [Perceptron training](perceptron-training.md) for the split and evaluation workflow.

## Run

```bash
py -3 -m pip install -r requirements.txt
py -3 -m unittest discover -s tests -v
py -3 -m src.cli --help
py -3 -m src.cli analyze --help
py -3 -m src.cli train-perceptron --help
```

Analyze explicitly selected CSV files or a directory containing the patient CSV families:

```powershell
py -3 -m src.cli analyze "C:\path\to\patient-csvs" --output "C:\path\to\results"
```

`--outcome-column` can select or prioritize a direct postoperative outcome column. Exact category lists can be overridden with `--success-values` and `--failure-values` after confirming the cohort's coding. Otherwise, configured columns and category rules embedded in `config/pipeline.xml` are used. Do not pass a broad folder containing unrelated CSVs: recursive discovery treats every CSV found as a patient file.

### Patient filename families

Files in the **same directory** with a base name plus a final numeric part are processed together: `ref_0002.csv`, `ref_0002_1.csv`, and `ref_0002_2.csv` represent one patient, while `ref_0003.csv` is another. A zero-padded suffix of three or more digits (such as `_0002`) is kept as part of the patient identifier; short parts such as `_1`, `_2`, and `_01` are treated as numbered continuations. `case_1.csv` and `case_2.csv` therefore also form one family even if `case.csv` is absent. Identical names in different directories are **not** merged. This is a filename convention, not verification of clinical identity; check ambiguous names before processing.

Each file is decoded with its own CSV header and delimiter. For one family, fields are aligned by header name, the earliest start and first subsequent end event are found across all files, and pre-operative features and postoperative evidence are selected from their respective time windows. The workflow assumes one index operation per filename family; multiple separate operations need an explicit episode key before clinical interpretation. Files and rows are processed in sorted file order; overlapping/duplicated records are **not deduplicated**, so verify the source exports before using sums or timing-dependent summaries. Header inventory remains per-file, while the patient-level analysis writes one vector and outcome per family. The `extraction_report.md` and logs report both input-file and patient-family counts without exposing filenames.

Generate one histogram/bar plot per mapped parameter, across the configured phases:

```powershell
py -3 -m src.cli distributions "C:\path\to\patient-csvs" --output "C:\path\to\results" --phase all --bins 20
```

The `distributions` command also writes the standard analysis artifacts. Distribution output consists of per-parameter plots under `parameter_distributions/`, `parameter_distribution_summary.csv`, and `parameter_distribution_categories.csv`. Parameters with no mapped values receive an explanatory empty plot.

Patient families are ordered by their first sorted file path before assigning anonymous IDs (`patient_001`, `patient_002`, …). Output tables and logs omit original filenames and paths. To relate pseudonyms to sources locally, reconstruct the same sorted family list; the mapping is not written by the app. Keep clinical outputs and logs local and out of version control.

## Feature vectors and targets

`patient_vectors.csv` has one row per filename family: `patient_id` and the mapped pre-operative dimensions, with **no label column**. `patient_outcomes.csv` stores the postoperative-derived target, status, and label rationale using the same patient ID; it contains no outcome score. The trainer joins these files by ID; it does not use row order or include outcomes as predictors. Numeric/categorical encodings are fitted from the training partition only.

The operation start is the earliest timestamped row whose event field contains any phrase in the **pre-operative item's `end_event`** under XML `included_phases` (defaults include `Start anesthesie`, `Start anesthesia`, `Start anaesthesia`, `Start surgery`, or `incision`). The end is the first later timestamped row matching any phrase in the **during-operative item's `end_event`** (`Afronden zorg`, `Stop anesthesie`, or `Vertrek naar IC (of_Afdeling)` by default). Configure each event's **CSV headers** with `event_fields` and its cell text with `phrases` in `config/pipeline.xml`, not in Python. Pre-operative rows are before start; during rows run from start up to but not including end; postoperative rows include end and later. Missing end means no during or timestamp-scoped postoperative evidence; explicit patient-scope metadata remains available. See [operation boundaries](xml-pipeline-configuration.md#editable-operation-boundaries). Numeric `unixtime` is treated as seconds (or milliseconds when its magnitude indicates milliseconds); ISO timestamps are also supported. Time-varying features use the last valid mapped reading strictly before the cutoff. If no cutoff exists, static demographics/operation-diagnosis proxy values are used only when there is exactly one distinct nonmissing value; time-varying baseline lab features remain missing.

## Binary outcome targets

`patient_outcomes.csv` contains binary targets derived from configured postoperative criteria: **-1 for failure**, **+1 for success**. Missing or unmatched evidence is `NA` and excluded from fitting/validation. Failure evidence takes precedence when rules disagree. The AKI-stage and mortality-flag criteria are provisional; a separate user-added recorded-death-days criterion marks any nonnegative numeric value, including `0`, as failure without claiming a 30-day endpoint. The other eight parameter criteria remain pending clinical definition. Categorical operation-result mappings are also configurable. See [Post-operative outcome rules](postoperative-outcomes.md).

Numeric features export only the numeric component before the first `|`; trailing annotations/components are discarded so measurements remain numeric. Other feature values remain as recorded. Each post-operative `<parameter>` in `config/pipeline.xml` owns its `measurement` entry with aggregation (`first`, `last`, `min`, `max`, `mean`, `sum`, or boolean `any_true`) and optional postoperative time window in hours. A measurement normally inherits its parameter's source fields; an explicit empty list means no measurement source. By default measurements require a timestamp on or after the operation **end**; `record_scope=patient` instead reads patient-level outcome metadata from any row, including a preoperative row. That metadata is never a model input. The default urine summary is the maximum recorded value, not a normalized urine-output criterion. These outputs are not clinical decision support.

## Phase-relative time series

Set `<entry name="export_time_series"><boolean>true</boolean></entry>` under the active XML's `<relevantParameters><settings>` (disabled in the starter template; enabled in the checked-in active configuration). Only selected during/post-operative `time_series_numeric` parameter readings appear in `time_series.csv` with `patient_id,phase,parameter_id,t_seconds,value`; no calendar dates or absolute timestamps are exported. The during clock starts at the configured operation-start event (`t_seconds=0`), and the postoperative-care clock resets at the configured operation-end event (`t_seconds=0`). A first measurement later than a marker retains its positive elapsed time; none is fabricated at zero. Missing markers/timestamps and missing or nonnumeric values produce no series row. Postoperative series obey the parameter's measurement source and optional `window_hours` limit. Numbered sibling files share the same clock. This export does not change aggregated measurements, outcome labels, or training inputs. See [phase-relative time-series export](xml-pipeline-configuration.md#phase-relative-time-series-export).

Run `py -3 -m src.cli plot-time-series` after exporting to make one interactive HTML plot per parameter for all patients. The CSV retains elapsed `t_seconds`, but the Plotly charts display elapsed hours (for example, a 72-hour post-operative window ends at 72 h). Use `--group-by class` for one multi-panel HTML plot per XML class, `--parameter id1,id2` and/or `--class epic.LAB` to filter parameters, and `--patient patient_001,patient_002` to select or compare patients; see [time-series plotting](xml-pipeline-configuration.md#time-series-plotting).

## Perioperative patient timeline dashboard

The active XML setting `preprocessing/plotting/patientTimeline/run` creates a patient-selectable Plotly dashboard. Its large numeric plot uses fields belonging to configured `<during><availableClasses>` prefixes **except classes marked untimed**, against operation-relative hours. Choose fields in the searchable Time series dropdown; check individual fields or Select all, and retain the subset when switching patients. The separate main-events timeline below combines operation start/end and postoperative-section-end boundaries with recorded pre-, during-, and post-operative events in phase lanes; records beyond the chosen end appear separately as later follow-up. XML-configured stay durations, including operation-end to first matching post-operative-end, are shown next to the boundary markers. Click an event for its phase, relative hour, and mapped timed values from the same row. Long phase lists render incrementally. `epic.DEMOGRAPHICS` values appear in a no-timestamp panel, not on the timeline. Absolute dates/timestamps are not included. Each numeric series is min–max normalized independently for visual comparison across unlike units; hover exposes the original raw value. With `compareAllPatients=true`, the dashboard opens with all patients' series overlaid and cohort tables that show patient pseudonyms side by side; select one patient in the header to inspect their timeline. `includeStudyPlots=true` adds collapsible in-dashboard embeds for generated local Plotly heatmaps, missingness plots, parameter distributions, time-series charts, and model evaluation plots when available. `preprocessing/plotting/theme` selects `dark` or `light` for the dashboard and generated plots. Separate pre-operative and post-operative tables and the overall summary remain available. The summary reports operation boundaries, duration, counts, missing timing, death status, and recorded days from operation to death when available. Hospital-after-operation duration is calculated from operation end to the configured post-operative-section end; ICU duration is calculated from configured ICU admission/discharge intervals clipped to the configured episode window, with overlaps merged. Duration rules, event phrases, fields, units, precision, and timezone are in the preprocessing XML. The checked-in ICU source fields, ISO 8601 format, Europe/Amsterdam timezone, and discharge phrases are candidates that must be confirmed against the data dictionary/event vocabulary before interpreting a real cohort. The death-day value comes from `epic.DEMOGRAPHICS.days mortality` and is displayed as recorded data, not calculated from timestamps; its reference point, units, and follow-up meaning should be confirmed before clinical interpretation. Missing death evidence is shown as not recorded, not as proof of survival. Missing boundary times are never fabricated. The template disables the dashboard. Leave `patientIds` empty to include all discovered patient families, or list pseudonyms to limit it. The HTML contains patient-level values; keep it and its local `plotly.min.js` bundle in approved storage and out of version control. See [patient timeline dashboard](xml-pipeline-configuration.md#patient-timeline-dashboard).

## Pre-operative and missingness heatmaps

The pre-operative and post-operative feature heatmaps use a deterministic sample of at most 500 patients; they are not cleanest-subset views. Numeric/categorical transformations are visualization-only. `preoperative_missingness_heatmap.html` is the default Plotly heatmap of missingness by pseudonymous patient: rows are patients, columns are selected pre-operative features, `1` means missing, and `0` means present. It is controlled by `preprocessing/plotting/missingnessByPatient` in the XML (both its `run` and parent `plotting/run` must be true); the active config enables it. No duplicate clean-subset or missingness-statistics CSV is produced.

## Post-operative heatmap

`postoperative_heatmap.html` includes every configured postoperative measurement ID for a deterministic patient sample; its aggregate values are also written to `postoperative_measurements.csv`. Each measurement's aggregation and optional time window are defined inside its owning post-operative parameter in `config/pipeline.xml`. Numeric values use the same per-parameter min–max formula as the pre-operative feature heatmap; constant columns are `0.5`, and missing/unresolved values are gray. Scaling is based on the plotted sample, not the raw clinical scale. Parameters with unresolved sources remain visible as empty columns rather than being silently omitted or fabricated. Time-series measurements require an identifiable operation end and timestamped postoperative rows.

## Missingness

Missing values stay blank except BMI, which is derived from available pre-operative weight and height when the BMI field is absent. Other missing selected inputs are not imputed; the trainer excludes those rows. The missingness heatmap is a visual overview only; no missingness-statistics CSV is generated.

## Output files

`--output results` creates three base CSVs (plus one if `<during>` has defined parameters, and one when phase-relative time-series export is enabled), plus plots/report. Existing files from older runs are left untouched; use a fresh output directory to see only the new compact set.

- `patient_vectors.csv` — patient ID plus pre-operative input features only; no labels.
- `patient_outcomes.csv` — patient ID, target (`-1`, `+1`, or `NA`), target status, and label basis from post-operative evidence.
- `duringoperative_measurements.csv` — optional, written when `<during>` defines parameters; latest mapped value in the `[start, end)` window per patient; never a model input.
- `postoperative_measurements.csv` — aggregated post-operative evidence for audit and distributions; never a model input.
- `time_series.csv` — optional phase-relative numeric readings with `t_seconds` and no date, created only when XML `export_time_series` is `true`; never a model input.
- `training_set.csv` and `validation_set.csv` are created by `train-perceptron`: training has the true target; validation has a `prediction` but no true target.
- `preoperative_heatmap.html`, `preoperative_missingness_heatmap.html`, `postoperative_heatmap.html`, `patient_timeline_dashboard.html`, and `extraction_report.md`.

Automated-test fixtures are synthetic; they do not validate the implementation on a clinical cohort. Keep cohort CSV files and patient-level outputs outside version control.
