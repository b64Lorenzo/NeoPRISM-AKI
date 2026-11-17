# XML pipeline configuration

`config/pipeline.xml` is the single runtime configuration. `config/pipeline.template.xml` is a disabled starter; `config/pipeline.xsd` validates both files and standalone `<parameterMap>` documents. No workbook or source-sheet file is required.

## Structure and execution

```xml
<pipelineConfiguration version="2">
  <logging>...</logging>
  <!-- Omit preprocessing to run directly from model-owned clean data. -->
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
  <models>
    <model id="perceptron-baseline" type="perceptron">
      <run>false</run>
      <mode>train-validation</mode>
      <trainingValidation>
        <vectors>data/clean/baseline/patient_vectors.csv</vectors>
        <outcomes>data/clean/baseline/patient_outcomes.csv</outcomes>
        <features><feature id="age" type="numeric" /></features>
        ...
      </trainingValidation>
      ...
    </model>
  </models>
</pipelineConfiguration>
```

This is an outline; use the complete template for a runnable file. Relative paths resolve against the project root, the parent of `config/`; keep alternate configurations in that directory. The CLI exposes only the `run` subcommand; `--config` selects the XML file and defaults to `config/pipeline.xml`. There are no standalone `analyze`, `distributions`, `plot-time-series`, or `train-perceptron` commands. Version 2 allows the whole `<preprocessing>` or `<models>` section to be omitted. A missing preprocessing section skips preprocessing; `<preprocessing><run>false</run>` skips raw extraction/reports, while an explicitly enabled time-series plot may still read an existing `time_series.csv`. Configured models still run from their own input paths. A missing or empty `<models>` section means no models run.

The runner initializes logging, runs preprocessing when present and enabled—or an explicitly enabled time-series plot from an existing CSV when extraction is disabled—then runs each enabled model entry sequentially in XML order. The first model error stops the pipeline. The patient timeline is a preprocessing plot, so it can embed completed study plots but not model plots generated later. A component's XSD declaration does not implement execution.

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

When `<preprocessing>` is present and its top-level `run` is true, the pipeline reads source CSVs and writes patient vectors, outcomes, and measurements. `parameterDescriptions/run` controls the definition-only Markdown reference, and `csvHeaderDescriptions/run` controls the header inventory. `plotting/run` controls preprocessing heatmaps, patient missingness, and distributions; it does not gate extraction. If preprocessing is omitted or disabled, raw extraction and these reports are skipped. A separately enabled time-series plot can still consume an existing time-series CSV when preprocessing is disabled. Models never receive implicit preprocessing output paths: configure each model's inputs explicitly, including when those paths point to preprocessing outputs.

### CSV header descriptions

`csvHeaderDescriptions/run` scans only each CSV's first logical record, independently of extraction. `inventoryOutput` records exact per-file column positions; `summaryOutput` records schemas, numbered-file families, header frequencies, phase/class assignments, unassigned headers and every discovered Epic class. Phase assignment comes from exact configured fields, not inferred class names. Numbered families remain directory-local, and the output directory is excluded from discovery.

Reports can contain identifying filenames even though no patient rows are read. Keep them local, and use a text-aware CSV viewer rather than a spreadsheet that might execute untrusted header formulas. Logs contain aggregate counts only.

### Distribution and missingness plot selection

The `distributions` and `missingnessByPatient` pipeline stages each have their own `run`, `phase`, `parameterIds/id` and `classNames/name` settings. `phase` accepts `all`, `preoperative`, `duringoperative` or `postoperative`; populated filters intersect, and empty lists mean all eligible mapped parameters in the chosen phase. Classes expand only their explicitly mapped XML members, not arbitrary raw CSV fields. Unknown names and empty explicit intersections are rejected. Plot selections never change extraction, outcome criteria or model inputs. These settings are not exposed as CLI subcommands or flags.

```xml
<distributions>
  <run>true</run>
  <bins>20</bins>
  <phase>all</phase>
  <parameterIds />
  <classNames />
</distributions>
<missingnessByPatient>
  <run>true</run>
  <phase>duringoperative</phase>
  <parameterIds />
  <classNames><name>epic.LAB</name></classNames>
  <output>results/selected_missingness.html</output>
</missingnessByPatient>
```

To migrate an earlier config, replace the boolean `<distributions>` and parent-level `<phase>`/`<bins>` with the nested distribution settings above. Add `<phase>preoperative</phase>`, `<parameterIds />` and `<classNames />` under an existing `missingnessByPatient` element; keep its current `run` and `output` values.

Set `preprocessing/run` to `true` to run extraction; set `preprocessing/plotting/run` to `true` to enable heatmaps, missingness, and distributions, then enable each child as needed. Distributions use a positive `bins` count and patient-level CSV tables; they write plots under `parameter_distributions/` plus `parameter_distribution_summary.csv` and `parameter_distribution_categories.csv`. Missingness writes to its configured `output` (default `results/preoperative_missingness_heatmap.html`) and uses the selected phases' extracted observations (`1` missing, `0` present), sampled deterministically to at most 500 patients. Valid zero values are present, not missing. By default, distributions include all eligible mapped parameters and missingness includes pre-operative features. `theme` accepts `dark` or `light` across Plotly outputs.

### Phase-relative time-series export

`parameters/settings/exportTimeSeries=true` writes `time_series.csv` with `patient_id,phase,parameter_id,t_seconds,value`. Only selected during/post `time_series_numeric` parameters are exported, using the configured measurement source fields/windows. During seconds start at operation start; post seconds start at operation end, not at the first observation. No synthetic zero-time reading, absolute timestamp, outcome metadata or model input is added. Readings without required timing/numeric values are skipped; configured 72-hour limits include their endpoint. This CSV is for review, not classifier input.

### Time-series plotting

`plotting/timeSeries/run` is independent of the parent plotting switch and can run when preprocessing is disabled, provided the configured output directory already contains `time_series.csv`. It consumes that CSV and never reopens raw cohort files. When preprocessing is enabled, `exportTimeSeries` must also be true to create the input CSV during extraction. `groupBy` accepts `parameter` or `class`; empty `parameterIds`, `classNames`, `patientIds` means all eligible items, and populated filters intersect. Class plots keep unlike units on separate axes. The CSV stays in seconds; charts display elapsed hours. Configure these filters in XML, then use the `run --config ...` command shown above; there is no standalone time-series plotting command.

### Patient timeline dashboard

`plotting/patientTimeline/run` enables the local HTML dashboard when `preprocessing/run` is true; it does not depend on the parent plotting switch. Empty `patientIds` means all families; `compareAllPatients` chooses the initial comparison view, and `includeStudyPlots` embeds completed study charts lazily. `layout/chartHeight` is 320–900 pixels; `panelMaxHeight` is 120–900. Both remain typed and configurable.

`duringOperative/availableClasses` preserves the 94-prefix visualization inventory; it does not add predictors. Untimed classes are excluded from numeric operation traces. Numeric fields are independently normalized for display with original values on hover. The sticky patient/series selectors preserve selected traces; a separate phase-aware event timeline exposes mapped row values and relative hours. Long event lists load in batches; pre/post/summary tables render when opened. Untimed demographics have their own panel. Patient-level values are embedded: keep the HTML and adjacent offline `plotly.min.js` bundle in approved storage.

## Models

`models/model` is a repeatable model instance. Its required `id` is unique within the pipeline, `type` selects a registered parser/runner, and its `run` child enables or skips that instance. Duplicate algorithms are allowed with different IDs; give each instance distinct output paths. The currently implemented type is `perceptron`. If `<models>` is absent or empty, the pipeline runs no model. Each model owns its clean-data inputs and outputs; the runner does not replace them with preprocessing output paths.

- `mode` is `train-validation` or `novel-data`.
- `trainingValidation` sets this model's vector/outcome inputs, output sets/model, train fraction, epochs, seed, `omit/id`, and a required `<features>` list. Each `<feature id="..." type="numeric|categorical"/>` must describe a predictor column in that model's clean vector CSV. This replaces the preprocessing parameter map as the trainer's feature/type source. The default omission remains `operation_diagnosis_proxy`; configure exclusions in this model's XML.
- `novelData` sets feature-only vectors, saved-model input and prediction output. Predictions contain no true target.
- `plotting/run` writes a validation confusion matrix and is valid only in train-validation mode.

For pre-clean data, omit `<preprocessing>` or set its top-level `run` false, then configure each model's vector/outcome paths and feature schema. To train from files produced by preprocessing, set those same model-owned paths explicitly to the generated files. Training joins by patient ID, fits encoders on training data only, and validates using prediction-only holdout exports. Labels are strictly `-1`/`+1`; `NA` is excluded rather than treated as a third class. A holdout result is not a calibrated probability.

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

## Version 1 to version 2 migration

Change the root `version` from `1` to `2`. Replace the singular `<model><activeModel>perceptron</activeModel><perceptron>...</perceptron></model>` with `<models><model id="perceptron" type="perceptron">...</model></models>`. Move the old perceptron `<run>` to the outer `<model>` entry; keep `mode`, training/validation, novel-data, and plotting settings inside it. Add `trainingValidation/features` entries for the clean vector columns and their types. The preprocessing block may be omitted; if retained with `run=false`, its child stages are skipped and model inputs still come from their configured paths. Both checked-in XML files use version 2.

## Migrating older local configurations

Replace `<inputs><path>...</path></inputs>` with one `<inputDirectory>...</inputDirectory>`. Move selection's run switch to `settings/selectionEnabled`, phase choices to phase `selected` attributes, and IDs to their phase's optional `includeIds`/`excludeIds`. Rename `pre`, `during`, `post` to `preOperative`, `duringOperative`, `postOperative`. Move each old included-phase end event to that phase's `endEvent/fields/field` and `endEvent/phrases/phrase`. Shared scalars/lists use the named settings above. Remove the old selection node, duplicated phase lists and workbook metadata. Both checked-in configurations are already migrated. Standalone `<parameterMap>` retains its typed-object format.
