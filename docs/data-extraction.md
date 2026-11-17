# CSV extraction

This guide describes patient-level outputs produced by the XML-configured pipeline. The only CLI subcommand is `run`; input/output paths and enabled stages come from the selected XML configuration. The historical row-oriented outputs are not produced by this workflow; see [patient-vector workflow](patient-vectors.md) for grouping, boundaries, and artifact details.

## Run configured extraction

Review `preprocessing/inputDirectory`, `preprocessing/outputDirectory`, and `preprocessing/run` in `config/pipeline.xml`. The input directory is searched recursively, so point it at a folder containing only the intended cohort. Then run:

```powershell
py -3 -m src.cli run --config config/pipeline.xml
```

The `--config` option may point to another pipeline XML file; when omitted, the CLI uses `config/pipeline.xml`. Same-directory numbered parts are grouped into one patient family; the naming rules are documented in the [patient-vector guide](patient-vectors.md). IDs such as `patient_001` are run-local pseudonyms, not stable identifiers. Original filenames are omitted from patient-level outputs and logs.

Core files are `patient_vectors.csv`, `patient_outcomes.csv`, `postoperative_measurements.csv`, and `extraction_report.md`; configured during-operative parameters add `duringoperative_measurements.csv`, and `exportTimeSeries=true` adds `time_series.csv`. `patient_vectors.csv` contains pre-operative features only. Postoperative evidence and labels remain separate from model inputs. Enabled plotting stages add HTML artifacts; see the [output list](patient-vectors.md#output-files).

## Optional distributions and time-series plots

Plots are pipeline stages configured in XML, not standalone CLI commands. Set `preprocessing/run` to `true` for extraction. `preprocessing/plotting/run` separately enables heatmaps, missingness, and distributions; enable the relevant child stages and configure their filters there. Distribution plots are written under `parameter_distributions/`, with `parameter_distribution_summary.csv` and `parameter_distribution_categories.csv`.

For time-series charts generated from the cohort, enable `preprocessing/run`, `preprocessing/parameters/settings/exportTimeSeries`, and `preprocessing/plotting/timeSeries/run`. The plotting stage reads `time_series.csv` rather than reopening patient CSVs; when extraction is disabled, it can still use an existing CSV. Set grouping and filters in XML. The CSV stores elapsed seconds; charts display elapsed hours. Plotly HTML can be viewed offline when its adjacent `plotly.min.js` bundle is kept with it. See the [XML guide](xml-pipeline-configuration.md#phase-relative-time-series-export) for export and plotting options. Run the configured stages with `py -3 -m src.cli run --config config/pipeline.xml`.

## Labels and scope

Targets are `-1` (failure) and `+1` (success); `NA` means evidence is missing, unmatched, or not yet defined. Failure evidence takes precedence. Several checked-in criteria are provisional or pending clinical definition; review the [outcome guide](postoperative-outcomes.md) before changing them. Automated fixtures are synthetic and do not establish clinical validity.

The preprocessing stage uses `src.preprocessing.patient_pipeline.analyze_inputs`. The separate `src.preprocessing.row_pipeline.analyze_inputs` remains for historical reference/tests; its old output descriptions are not instructions for the current workflow. Keep source data and generated patient-level artifacts in approved storage, outside version control.