# XML pipeline configuration

`config/pipeline.xml` is the single runtime configuration. `config/pipeline.template.xml` is a disabled starter; `config/pipeline.xsd` validates both files and standalone `<parameterMap>` documents. No workbook or source-sheet file is required.

## Structure and execution

```xml
<pipelineConfiguration version="1">
  <logging>...</logging>
  <preprocessing>
    <run>false</run>
    <inputDirectory>data/patients</inputDirectory>
    <outputDirectory>results</outputDirectory>
    <parameters formatVersion="1">
      <settings>...</settings>
      <preOperative selected="true">...</preOperative>
      <duringOperative selected="true">...</duringOperative>
      <postOperative selected="true">...</postOperative>
    </parameters>
    <parameterDescriptions>...</parameterDescriptions>
    <csvHeaderDescriptions>...</csvHeaderDescriptions>
    <plotting>...</plotting>
  </preprocessing>
  <model>...</model>
</pipelineConfiguration>
```

This is an outline; use the complete template for a runnable file. Relative paths resolve against the project root, the parent of `config/`; keep alternate configurations in that directory. The XML accepts one `inputDirectory`; direct `analyze` and `distributions` commands still accept multiple CSV files/directories.

The runner initializes logging, writes enabled definition/header reports, extracts patients, generates preprocessing/time-series plots, executes the selected model, then builds the patient dashboard so it can embed completed study plots. A component's XSD declaration does not implement execution.

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m src.cli run --config config/pipeline.xml
```

**Review inputDirectory and run switches before running the active configuration:** it may point outside the repository. Tests use isolated synthetic projects, never the configured cohort. Keep inputs, generated patient values and logs in approved storage, outside Git.

## Shared settings

`logging/directory` and `logging/level` configure per-run logs; supported levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`.

The parameter settings have a fixed, compact order:

```xml
<settings>
  <selectionEnabled>false</selectionEnabled>
  <exportTimeSeries>false</exportTimeSeries>
  <missingValueTokens><token/><token>na</token><token>n/a</token><token>nan</token><token>null</token></missingValueTokens>
  <derivedFeatures><object>...</object></derivedFeatures>
  <outcomeRules><object>...</object></outcomeRules>
</settings>
```

Complex rule objects retain typed entries (`string`, `number`, `boolean`, `null`, `object`, `array`), so unknown units remain null and numbers/booleans keep their types. `derivedFeatures` holds BMI assumptions; verify kg/cm against the actual data dictionary. `outcomeRules` contains explicit direct outcome categories. Parameter-specific success/failure criteria remain on their owning parameter. Raw laboratory values are observations and do not automatically generate outcome labels. No workbook provenance, workbook rows, source sheets, duplicated phase lists or unused catalog-description tables are required.

### Phase-owned definitions and selection

The required parameter-section order is **settings, preOperative, duringOperative, postOperative**. Each phase contains its `endEvent`, optional `includeIds`/`excludeIds`, then its parameter/class definitions; `duringOperative` may put `availableClasses` before those definitions. Every definition occurs once. Class groups contain only explicitly defined matching `epic.CLASS.item` fields, never arbitrary raw CSV columns.

`settings/selectionEnabled=false` ignores all phase filters. When true:

- `selected="false"` skips a phase; omitted `selected` means true.
- Optional `<includeIds><id>age</id></includeIds>` narrows **that phase only**. Omitted/empty includes all its definitions.
- Optional `<excludeIds><id>...</id></excludeIds>` removes IDs from that phase.
- IDs must belong to their owning phase, and include/exclude lists must not overlap. Select at least one phase.

The catalog loader always returns all definitions. The runner applies enabled selection to extraction, reference documents and plots; header inventory uses the complete defined catalog. There is no separate `<selection>` node or selection Python package.

Post-operative `measurement` and `outcome_criteria` entries live inside their owning parameter. Measurements inherit source fields when omitted; an explicit empty array means no source. `measurementOrder` preserves output order. `record_scope=patient` permits patient-level outcome metadata from any row, but cannot specify a postoperative time window and never makes that value a predictor. Optional post-operative class `failure="nonempty"` treats any selected non-missing value, including `0`/`false`, as failure; enable only after review. It is disabled by default.

The current catalog has 9 pre-operative, 1 during-operative and 10 post-operative definitions. Preserve the separation between pre-operative model inputs and during/post-operative review/outcome data. `recorded_death_days` records death for nonnegative values including zero, but its follow-up horizon is unknown; it is not a verified 30-day endpoint. Missing or undefined criteria remain unclassified, not a third training class.

## Editable operation boundaries

Each phase owns a normal XML `endEvent`, rather than repeating phase objects under shared settings:

```xml
<preOperative selected="true">
  <endEvent>
    <fields><field>epic.EVENTS.EVENT</field><field>event</field><field>events</field></fields>
    <phrases><phrase>start surgery</phrase></phrases>
  </endEvent>
  <includeIds><id>age</id></includeIds>
  <!-- Explicit parameter/class definitions follow. -->
</preOperative>
```

`preOperative/endEvent` marks operation start; `duringOperative/endEvent` marks operation end; `postOperative/endEvent` marks the postoperative stay endpoint. `fields` contains exact CSV headers and `phrases` contains event-cell text; matching is case-insensitive and whitespace-normalized. The checked-in files retain all configured start/end phrases and may use different source columns for each boundary.

Across a numbered-file patient family, choose the earliest timestamped start, the earliest end strictly after it, and the earliest configured postoperative-end marker strictly after operation end. Pre-operative values are before start; during values are in `[start, end)`; timestamp-scoped post-operative evidence starts at end. Missing markers never cause evidence to be assigned to the wrong phase. Without a start, time-varying pre-operative values stay missing except explicitly configured static fallback. Without end, during/post timestamp-scoped values remain missing; patient-scope outcomes can still apply. Multiple independent operations within one family are not split into separate episodes.

`preOperative/class/timed=false` identifies untimed metadata. Extraction still enforces a known dated pre-operative cutoff; otherwise exactly one distinct non-missing family value is required. The dashboard displays untimed values separately, without fabricated timestamps.

### XML-configured duration summaries

Hospital duration uses operation end through the earliest configured postoperative-end event. ICU duration parses the configured admission/discharge fields, clips and merges overlapping intervals in the configured operation/stay window, and sums them. Methods, boundary/source names, units, precision, ISO 8601 datetime handling and timezone remain explicit parameter measurement settings. Missing source values or patient boundary markers display unavailable; no stay duration is fabricated.

The active discharge phrases and ICU datetime fields/timezone are candidates requiring source confirmation. A signed visit alone does not prove discharge. Events after the chosen postoperative endpoint appear in a follow-up timeline lane; they do not silently extend the stay.

## Preprocessing reports and plotting

`parameterDescriptions/run` writes a definition-only Markdown reference. `preprocessing/run` controls source-row extraction. Disabled preprocessing can still run independently enabled header inventories or plots against existing output files.

### CSV header descriptions

`csvHeaderDescriptions/run` scans only each CSV's first logical record, independently of extraction. `inventoryOutput` records exact per-file column positions; `summaryOutput` records schemas, numbered-file families, header frequencies, phase/class assignments, unassigned headers and every discovered Epic class. Phase assignment comes from exact configured fields, not inferred class names. Numbered families remain directory-local, and the output directory is excluded from discovery.

Reports can contain identifying filenames even though no patient rows are read. Keep them local, and use a text-aware CSV viewer rather than a spreadsheet that might execute untrusted header formulas. Logs contain aggregate counts only.

### Patient-by-feature missingness heatmap

Enable both `plotting/run` and `plotting/missingnessByPatient/run`. The configured output is an interactive patient-by-pre-operative-feature map (`1` missing, `0` present), sampled deterministically to at most 500 patients. Heatmap scaling/encoding is visual only; exported feature values are unchanged. `plotting/distributions` additionally enables distributions; `phase` and positive `bins` control those plots. `theme` accepts `dark` or `light` across Plotly outputs.

### Phase-relative time-series export

`parameters/settings/exportTimeSeries=true` writes `time_series.csv` with `patient_id,phase,parameter_id,t_seconds,value`. Only selected during/post `time_series_numeric` parameters are exported, using the configured measurement source fields/windows. During seconds start at operation start; post seconds start at operation end, not at the first observation. No synthetic zero-time reading, absolute timestamp, outcome metadata or model input is added. Readings without required timing/numeric values are skipped; configured 72-hour limits include their endpoint. This CSV is for review, not classifier input.

### Time-series plotting

`plotting/timeSeries/run` is independent of the parent heatmap switch. It consumes existing/preprocessed `time_series.csv`, never reopens raw cohort files. With preprocessing enabled, `exportTimeSeries` must also be true. `groupBy` accepts `parameter` or `class`; empty `parameterIds`, `classNames`, `patientIds` means all eligible items, and populated filters intersect. Class plots keep unlike units on separate axes. The CSV stays in seconds; charts display elapsed hours.

```powershell
py -3 -m src.cli plot-time-series --group-by class --class epic.LAB --parameter during_creatinine --patient patient_001,patient_002
```

### Patient timeline dashboard

`plotting/patientTimeline/run` independently enables the local HTML dashboard. Empty `patientIds` means all families; `compareAllPatients` chooses the initial comparison view, and `includeStudyPlots` embeds completed study charts lazily. `layout/chartHeight` is 320–900 pixels; `panelMaxHeight` is 120–900. Both remain typed and configurable.

`duringOperative/availableClasses` preserves the 94-prefix visualization inventory; it does not add predictors. Untimed classes are excluded from numeric operation traces. Numeric fields are independently normalized for display with original values on hover. The sticky patient/series selectors preserve selected traces; a separate phase-aware event timeline exposes mapped row values and relative hours. Long event lists load in batches; pre/post/summary tables render when opened. Untimed demographics have their own panel. Patient-level values are embedded: keep the HTML and adjacent offline `plotly.min.js` bundle in approved storage.

## Models

`model/activeModel` selects a registered configuration class; perceptron is currently implemented. Each model owns its configuration and execution handler. Under `model/perceptron`:

- `run` enables execution; `mode` is `train-validation` or `novel-data`.
- `trainingValidation` sets vector/outcome inputs, output sets/model, train fraction, epochs, seed and `omit/id`. The default omission remains `operation_diagnosis_proxy`; CLI `--omit` adds exclusions. Newly preprocessed vectors/outcomes take precedence over saved input paths.
- `novelData` sets feature-only vectors, saved-model input and prediction output. Predictions contain no true target.
- `plotting/run` writes a validation confusion matrix and is valid only in train-validation mode.

Training joins by patient ID, fits encoders on training data only, and validates using prediction-only holdout exports. Labels are strictly `-1`/`+1`; `NA` is excluded rather than treated as a third class. A holdout result is not a calibrated probability.

## Editing and validating

Use [the XML editor guide](parameter-map-xml.md) and keep `pipeline.xsd` beside the XML. Validate structure and run synthetic tests after edits:

```text
xmllint --noout --schema config/pipeline.xsd config/pipeline.xml config/pipeline.template.xml
```

```powershell
py -3 -m unittest discover -s tests -v
```

Runtime validation also checks semantics such as known IDs, matching phase ownership, boundaries, unit assumptions and mode compatibility. XSD validity does not prove source availability or sufficient training cases.

## Extending the schema and pipeline

Create the component in its owning package; add and document its type/element in the single XSD, both XML examples, parser, execution path, guide and synthetic tests. Keep execution order deliberate. Add models under `src/models/<name>/`; register the parser in `MODEL_CONFIGS` and the handler in `MODEL_RUNNERS`. A parser alone is not a runnable model. See [code structure](code-structure.md).

## Migrating older local configurations

Replace `<inputs><path>...</path></inputs>` with one `<inputDirectory>...</inputDirectory>`. Move selection's run switch to `settings/selectionEnabled`, phase choices to phase `selected` attributes, and IDs to their phase's optional `includeIds`/`excludeIds`. Rename `pre`, `during`, `post` to `preOperative`, `duringOperative`, `postOperative`. Move each old included-phase end event to that phase's `endEvent/fields/field` and `endEvent/phrases/phrase`. Shared scalars/lists use the named settings above. Remove the old selection node, duplicated phase lists and workbook metadata. Both checked-in configurations are already migrated. Standalone `<parameterMap>` retains its typed-object format.
