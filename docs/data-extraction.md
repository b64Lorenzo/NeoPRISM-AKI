# CSV extraction: current implementation

This page describes the active `src.cli` workflow. Same-directory CSVs in one numbered filename family are treated as **one patient time series**; the old row-oriented extraction reports (`postoperative_labels.csv`, per-source `input_NNN` exports, and missingness-severity summaries) are not produced by the current CLI. For feature selection and outcome details, see [Patient-vector workflow](patient-vectors.md), [Pre-operative parameters](preoperative-parameters.md), and [Post-operative outcome rules](postoperative-outcomes.md).

## Analyze

```powershell
py -3 -m src.cli analyze "C:\path\to\patient-csvs" --output "C:\path\to\results"
```

Pass individual CSVs or directories. Directories are searched recursively for CSV files; use a folder containing only the intended cohort. Files in the same directory are grouped by the patient-family rule before assigning `patient_001`, `patient_002`, etc. `ref_0002.csv` and `ref_0002_1.csv` are one patient, but `ref_0003.csv` is another. Original filenames are not included in patient-level outputs or logs. These IDs are run-local pseudonyms, not stable identifiers across runs.

`config/pipeline.xml` is the runtime map: `<preprocessing><relevantParameters>` contains shared settings followed by definitions written directly under `<pre>`, `<during>`, and `<post>` (9, 1, and 11 respectively). A class groups explicitly defined matching fields, never all CSV headers. The post-operative `measurement` and `outcome_criteria` typed entries belong to each parameter, not to a repeated sibling catalog. The source workbook contains other phases but is not bundled. `recorded_death_days` is user-added from an exact CSV header, not the workbook. Pre-operative features remain separate from postoperative measurements and targets. During-operative definitions are extracted only between the XML-configured timestamped start/end events and written to `duringoperative_measurements.csv` when any such fields are defined. Each `included_phases` item defines its own `end_event` with editable `event_fields` and `phrases` (the post-operative end event is null); see [the XML guide](xml-pipeline-configuration.md#editable-operation-boundaries). See the [XML parameter-map guide](parameter-map-xml.md) for editing and validation.

## Outcome labels

The only trained labels are `-1` (failure) and `+1` (success). `NA` means no configured rule matched or required evidence was absent; it is not a class and is excluded from training/evaluation. When configured postoperative criteria disagree, a failure criterion wins. There are no reward scores, weighted aggregation, or severity bands.

Currently the AKI-stage rule maps stages 0–1 to success and 2–3 to failure, and the mortality flag maps `0` to success and `1` to failure. Both are provisional; the mortality horizon is unverified. The separate user-defined `recorded_death_days` rule maps any nonnegative number, including `0`, to failure, with no success inference from missing data and no assumed 30-day limit. The other eight postoperative parameter criteria are explicitly marked `pending_clinical_definition`, so those measurements do not independently classify patients. Configured direct operation-result categories can also provide labels (`optimal`/`mid` as success; `severe`/death categories as failure). Unknown categories remain unclassified. See the outcome guide before changing these clinical/project rules.

## Output files

`analyze` writes:

- `patient_vectors.csv` — patient ID and pre-operative features only.
- `patient_outcomes.csv` — patient ID, target, status, and label basis; no score column.
- `duringoperative_measurements.csv` — optional, for explicitly defined during-operative parameters; never a model input.
- `postoperative_measurements.csv` — aggregated post-operative values from the operation end onward, kept out of model inputs.
- `time_series.csv` — optional when XML `<settings>` has `export_time_series=true`: selected during/post-operative `time_series_numeric` readings with phase-relative elapsed `t_seconds`, not dates or absolute timestamps; see [the XML guide](xml-pipeline-configuration.md#phase-relative-time-series-export).
- `preoperative_heatmap.html`, `preoperative_missingness_heatmap.html`, `postoperative_heatmap.html`, and `extraction_report.md`.

The `distributions` command first creates the standard analysis outputs, then writes per-parameter interactive plots under `parameter_distributions/` and the summary CSVs (`parameter_distribution_summary.csv` and `parameter_distribution_categories.csv`) in the selected output directory.

All generated charts use Plotly interactive HTML and write one shared local `plotly.min.js` bundle in each chart directory; keep the bundle beside the HTML files to open them offline. Grafana is intentionally not part of this batch pipeline; it would add a separately hosted dashboard and data-source service.

The `plot-time-series` command reads only the generated `time_series.csv` and writes interactive HTML charts under `time_series_plots/`; it does not inspect the original patient CSVs. It plots all configured series by default, or filters/group them by XML class, parameter ID, or pseudonymous `patient_id`. The XML runner can invoke the same plotter through `preprocessing/plotting/timeSeries`, using either the file just exported or an existing file in `preprocessing/outputDirectory`. See [time-series plotting](xml-pipeline-configuration.md#time-series-plotting).

The legacy row-oriented `src.pipeline.analyze_inputs` remains only for reference/tests; `src.cli` uses the patient-level `src.patient_pipeline.analyze_inputs`. The two modules share file discovery and patient-family grouping helpers. Do not use the historical row-oriented output descriptions as instructions for the current workflow. All automated fixtures are synthetic; no clinical cohort validity is implied.
