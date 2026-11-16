# CSV extraction

This guide covers the current patient-level `src.cli` workflow. The historical row-oriented outputs are not produced by the CLI; see [patient-vector workflow](patient-vectors.md) for grouping, boundaries, and artifact details.

## Analyze a cohort

```powershell
py -3 -m src.cli analyze "C:\path\to\patient-csvs" --output "C:\path\to\results"
```

Pass CSV files or directories. Directory discovery is recursive, so use a folder containing only the intended cohort. Same-directory numbered parts are grouped into one patient family; the naming rules are documented in the [patient-vector guide](patient-vectors.md). IDs such as `patient_001` are run-local pseudonyms, not stable identifiers. Original filenames are omitted from patient-level outputs and logs.

Core files are `patient_vectors.csv`, `patient_outcomes.csv`, `postoperative_measurements.csv`, and `extraction_report.md`; configured during-operative parameters add `duringoperative_measurements.csv`, and `exportTimeSeries=true` adds `time_series.csv`. `patient_vectors.csv` contains pre-operative features only. Postoperative evidence and labels remain separate from model inputs. Enabled plotting stages add HTML artifacts; see the [output list](patient-vectors.md#output-files).

## Distributions and time-series plots

`distributions` runs the standard analysis and adds one interactive plot per mapped parameter under `parameter_distributions/`, plus `parameter_distribution_summary.csv` and `parameter_distribution_categories.csv`:

```powershell
py -3 -m src.cli distributions "C:\path\to\patient-csvs" --output "C:\path\to\results" --phase all --bins 20
```

After enabling time-series export in the XML configuration, plot its CSV:

```powershell
py -3 -m src.cli plot-time-series --input results/time_series.csv
```

`plot-time-series` reads only `time_series.csv`; it does not reopen patient CSVs. Use `--group-by class`, `--class`, `--parameter`, and `--patient` to filter plots. The CSV stores elapsed seconds; charts display elapsed hours. Plotly HTML can be viewed offline when its adjacent `plotly.min.js` bundle is kept with it. See the [XML guide](xml-pipeline-configuration.md#phase-relative-time-series-export) for export and plotting options.

## Labels and scope

Targets are `-1` (failure) and `+1` (success); `NA` means evidence is missing, unmatched, or not yet defined. Failure evidence takes precedence. Several checked-in criteria are provisional or pending clinical definition; review the [outcome guide](postoperative-outcomes.md) before changing them. Automated fixtures are synthetic and do not establish clinical validity.

The CLI uses `src.preprocessing.patient_pipeline.analyze_inputs`. The separate `src.preprocessing.row_pipeline.analyze_inputs` remains for historical reference/tests; its old output descriptions are not instructions for the current workflow. Keep source data and generated patient-level artifacts in approved storage, outside version control.