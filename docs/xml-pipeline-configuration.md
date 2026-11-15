# XML pipeline configuration

The project runs from `config/pipeline.xml`, validated against the **single** schema, `config/pipeline.xsd`. That schema describes the whole run configuration and the typed parameter catalog (both embedded `<relevantParameters>` and standalone `<parameterMap>` roots); there is no separate parameter-map XSD. The top-level groups map to the configuration sections, with preprocessing and model settings nested beneath them:

```xml
<pipelineConfiguration>
  <logging>...</logging>
  <preprocessing>
    <parameterSelection>...</parameterSelection>
    <parameterDescriptions>...</parameterDescriptions>
    <csvHeaderDescriptions>...</csvHeaderDescriptions>
    <plotting>...</plotting>
  </preprocessing>
  <model>
    <activeModel>perceptron</activeModel>
    <perceptron>
      <trainingValidation>...</trainingValidation>
      <novelData>...</novelData>
      <plotting>...</plotting>
    </perceptron>
  </model>
</pipelineConfiguration>
```

The real file also contains the values, explicit `run` switches, and schema reference. `config/pipeline.template.xml` is a schema-valid starter with execution switches disabled.

## Order and execution

The top-level order is intentional and validated by an XSD `xs:sequence`: `logging`, then `preprocessing`, then `model`. `<logging>` is a direct child of `<pipelineConfiguration>` and supplies shared settings; it is initialized for the run, not treated as a processing stage. The runner executes enabled processing stages in their XML order, so preprocessing outputs are ready before the model consumes them. Within each section, child elements also have a schema-defined order for consistent reading and editing.

When adding or moving a pipeline stage, change the top-level sequence in `config/pipeline.xsd`, place the matching node in that position in both XML files, and update the configuration parser and runner so the stage is executed at that point. Adding an XSD node alone only makes the XML valid; it does not implement or execute a component.

Run the configured workflow from the project root:

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m src.cli run
```

Or choose another XML file:

```powershell
py -3 -m src.cli run --config config/pipeline.xml
```

The starter template points to an empty `data/patients/` directory. The active `config/pipeline.xml` may point to an approved external cohort; **review its inputs and stage switches before running it**. A patient may have multiple numbered files in the same directory. Patient files, logs, and generated results are not source code and should not be committed.

## Configuration sections

### `logging` (direct child of `pipelineConfiguration`)

- `directory` sets the log output folder.
- `level` accepts `DEBUG`, `INFO`, `WARNING`, or `ERROR`.

### `preprocessing`

- `run` enables or skips CSV preprocessing.
- `inputs/path` lists one or more files or folders to scan.
- `outputDirectory` sets where patient vectors, outcomes, measurements, and reports are written.
- `relevantParameters` has `<settings>` for shared typed options followed by `<pre>`, `<during>`, and `<post>`. Each phase **owns its full parameter definitions**: `<parameter id="...">` contains metadata and exact source fields; `<class name="epic.CLASS">` groups matching defined parameters, without promoting unknown CSV columns. No sibling `<object>` contains a second catalog, and IDs/phases are not duplicated as typed entries. A source such as `epic.LAB.Kreatinine (umol/L)` may have distinct pre- and post-operative definitions, with no outcome leaking into predictors. `recorded_death_days` maps to `epic.DEMOGRAPHICS.days mortality` without assuming a 30-day endpoint.
- Each post-operative parameter owns its typed `outcome_criteria` and `measurement` entries. An optional `<post><class name="epic.CLASS" failure="nonempty">…defined parameters…</class></post>` treats **any nonblank, non-missing value** in selected, catalogued post-operative fields of that class as failure (`-1`), including `0` or `false`. Enable this broad rule only after clinical review; it is off in the checked-in XML. Pre-/during-operative classes cannot declare failure.
- Post-operative `measurement` entries configure aggregation and optional `window_hours`: the measurement inherits its parameter's source fields when absent, but an explicit empty source-field array means no source. `record_scope=patient` reads outcome metadata from any row; the default uses postoperative timestamps. Patient scope cannot specify `window_hours`. `measurementOrder` preserves the existing export-column order; new measurements without it follow in XML order. Patient-scope metadata is excluded from model inputs.
- `parameterSelection/run` enables a **second**, optional filter on the phase-owned definitions. When enabled, `phases` selects pre/during/post-operative phases; `includeIds` optionally narrows that set; `excludeIds` removes IDs. Empty `includeIds` means all defined parameters in those phases. The same filter is applied to preprocessing outputs, parameter documentation, and distributions; the CSV header report always uses the primary `<pre>/<during>/<post>` definitions, not this optional execution filter.
- `parameterDescriptions/run` writes a Markdown parameter reference without patient data.
- `csvHeaderDescriptions/run` independently scans the configured `inputs/path` CSVs and writes an exact, per-file header inventory plus a Markdown summary with filename families, ordered schemas, header frequency, phase-specific header/class tables, unassigned headers, and a final index of **all discovered Epic classes**. It works even when `preprocessing/run` is `false`.
- `plotting/run` controls preprocessing heatmaps; `distributions` additionally enables per-parameter plots. `phase` accepts `all`, `preoperative`, `duringoperative`, or `postoperative`; `bins` controls histogram resolution. `missingnessByPatient/run` selects the patient-by-feature missingness heatmap and `output` sets its Plotly `.html` path; both it and the parent `plotting/run` must be `true`. The active config enables this default plot, while the starter template disables it. The nested `timeSeries` component has its own `run` switch and can plot the generated or existing `time_series.csv` without enabling the preprocessing heatmaps.

The checked-in XML defines 9 pre-operative, 1 during-operative, and 11 post-operative parameters. The example `<during><class name="epic.LAB"><parameter id="during_creatinine">…` uses the configured `epic.LAB.Kreatinine (umol/L)` header; verify its availability, timing, and units before clinical use. `duringoperative_measurements.csv` records one last numeric value per patient from `[operation start, operation end)`. This export is enabled by the presence of an included during-operative definition and is **not** a perceptron input. Enable `export_time_series` separately for all selected individual readings with relative time. During-operative distributions can also be plotted. The default XML has no class-wide failure flags; only individually reviewed provisional postoperative rules are active.

#### Patient-by-feature missingness heatmap

The `preprocessing/plotting/missingnessByPatient` item controls the Plotly HTML heatmap of configured pre-operative features by pseudonymous patient. Set both `plotting/run` and its `run` child to `true`; the active `config/pipeline.xml` does so by default, and `config/pipeline.template.xml` leaves it disabled. `output` selects the `.html` destination. Rows are patients, columns are selected pre-operative features, red (`1`) means missing, and green (`0`) means present. For large cohorts the plot uses a deterministic sample of at most 500 patients. This is a diagnostic view only; it does not impute or modify model inputs.

```xml
<missingnessByPatient>
  <run>true</run>
  <output>results/preoperative_missingness_heatmap.html</output>
</missingnessByPatient>
```

#### Patient timeline dashboard

The `<patientTimeline>` child generates a local interactive Plotly dashboard with a pseudonymous patient selector. Its `run` switch is independent of the parent heatmap/distribution switch: the active `config/pipeline.xml` enables it and the starter template disables it. An empty `<patientIds />` includes every discovered patient family; add `<id>patient_001</id>` children to limit it. `<compareAllPatients>false</compareAllPatients>` is the checked-in default: it opens one patient and omits the cohort-table payload. Set it to `true` to start in cohort comparison mode; the header selector still switches to any patient. `includeStudyPlots` embeds generated Plotly study charts in collapsible sections. The output must be an `.html` file.

The `<relevantParameters><during><availableClasses>` inventory contains **94 Epic class prefixes** that may be considered for the operation plot. A class explicitly marked `<timed>false</timed>` (currently `epic.DEMOGRAPHICS`) is removed from the plotted subset even if listed in the inventory. The dashboard scans only columns matching the remaining configured prefixes; this inventory is visualization-only and does not add raw fields to model inputs. During-operation numeric columns are plotted against elapsed hours from the configured operation-start event and are independently min–max normalized for visual comparison across unlike units; hover shows each original field and raw value. Configured event-field values appear on the timeline for all three phases. Selecting a marker shows its phase, operation-relative elapsed time, and mapped values from that same row; embedded event data contains relative time, not absolute dates/timestamps. The all-patient view overlays patient series and shows pre-operative, post-operative, and summary cohort tables with pseudonymous IDs; the header selector switches to a single patient's full detail. `includeStudyPlots=true` adds available generated preprocessing, distribution, missingness, time-series, and model-evaluation HTML plots as collapsible embeds. The `theme` value under `<preprocessing><plotting>` accepts `dark` or `light` and is applied across Plotly outputs. Patient summary includes configured duration metrics: `hospital_los` is operation end through the configured post-operative-section end; `icu_los` is the configured ICU interval overlap within its configured episode window, merging overlapping intervals. Duration fields, boundary phrases, interval aggregation, units, precision, datetime format, and timezone are read from XML. The checked-in ICU source fields (`q_umcg_aki_marco.adm_ic_admdate` / `adm_ic_disdate`), ISO 8601 format, Europe/Amsterdam timezone, and discharge event phrases are candidates that must be confirmed against the data dictionary and event vocabulary before cohort use. The summary also reports death status and recorded operation-to-death distance when available; the value comes from `epic.DEMOGRAPHICS.days mortality`, not timestamp subtraction, so confirm its operation reference, units, and follow-up meaning before clinical interpretation. Missing death evidence is not interpreted as survival; missing boundaries are reported, not fabricated. Patient IDs are assigned from ordered input families and may change when the input set changes. The full class inventory belongs directly under `<relevantParameters><during>` and precedes explicit mapped definitions; both checked-in XML files contain the 94 exact Epic prefixes.

```xml
<during>
  <availableClasses>
    <className>epic.EVENTS</className>
    <className>epic.LAB</className>
    <!-- The active and template XML files list every configured class. -->
  </availableClasses>
  <!-- Explicitly mapped during-operative parameters remain separate. -->
</during>
```

```xml
<patientTimeline>
  <run>true</run>
  <patientIds />
  <compareAllPatients>false</compareAllPatients>
  <includeStudyPlots>true</includeStudyPlots>
  <layout>
    <chartHeight>740</chartHeight>
    <panelMaxHeight>320</panelMaxHeight>
  </layout>
  <output>results/patient_timeline_dashboard.html</output>
</patientTimeline>
```

Set the shared Plotly theme under `<preprocessing><plotting>`:

```xml
<plotting>
  <theme>dark</theme>
  <!-- other plot settings -->
</plotting>
```

The dashboard has one large Plotly plot for independently normalized during-operative numeric series. Its sticky header contains a patient selector and a searchable time-series dropdown: tick individual fields or use **Select all**; the checkbox reflects partial selection and selected fields persist across patient/cohort switches. Deselecting all only empties the numeric plot, not the event timeline. A single main-events timeline directly below the plot puts operation start/end and postoperative-section-end boundaries alongside recorded **pre-, during-, and post-operative events** in separate phase lanes, chronologically within each lane. It shows **relative hours** (or “Unavailable” for missing boundaries), not calendar dates; select an event to inspect its mapped measurements. Long event lists initially display 50 entries per phase with a Show more button, so the page stays responsive. When comparing all patients, select one patient to inspect their events and untimed demographics. `epic.DEMOGRAPHICS` remains in its own no-timestamp panel, not the time axis. The pre-operative, post-operative, and summary disclosure panels render tables only when opened and scroll independently; study-plot iframes load on demand. The reusable `DashboardLayout` reads `<layout>` from XML: `chartHeight` is 320–900 pixels (740 by default), and `panelMaxHeight` is 120–900 pixels (320 by default). Both are required positive integers in the XSD and checked against those ranges when the configuration loads. Edit them in both XML files to adjust chart and panel size without changing code.

The dashboard embeds patient-level values and event text. Keep its HTML and adjacent `plotly.min.js` in approved local storage; do not share or commit the files. It contains elapsed time, not absolute dates.

#### XML-configured duration summaries

Stay-duration calculations are stored on the corresponding measurement in `config/pipeline.xml`; no source field, boundary, unit, precision, datetime format, or timezone is embedded in dashboard logic. The configured hospital metric subtracts the operation-end timestamp from the **earliest configured post-operative-end event after operation end**, then converts to days. The main-events timeline displays this configured stay duration beside the boundaries; events later than that boundary appear in a separate “After post-operative end” lane rather than silently extending the stay. With `admission end` at +290.75 h and `Getekend bezoek` at +7399.02 h, the first event sets the boundary; the later visit does not extend the computed interval. These phrases are editable candidates, **not proof of hospital discharge**: a signed visit may not mean discharge, especially when it is the only matching marker. Confirm event semantics against the clinical source before interpreting the displayed duration as actual hospital length of stay. The ICU metric parses the two configured interval fields, clips each valid interval to the configured operation episode window, merges duplicate/overlapping spans, and sums the resulting overlap. Missing boundaries, missing or unparsable fields, and invalid intervals produce `Unavailable`; source datetimes are not displayed in the dashboard. For example, the duration child for the hospital metric is:

```xml
<entry name="duration"><object>
  <entry name="method"><string>phase_boundary_interval</string></entry>
  <entry name="aggregation"><string>difference</string></entry>
  <entry name="start_boundary"><string>operation_end</string></entry>
  <entry name="end_boundary"><string>postoperative_end</string></entry>
  <entry name="unit"><string>days</string></entry>
  <entry name="precision"><number>2</number></entry>
  <entry name="label"><string>Time in hospital after operation</string></entry>
</object></entry>
```

The active XML currently treats ICU date headers, ISO 8601 parsing, `Europe/Amsterdam`, and the discharge phrases as candidate values. Confirm those against the study data dictionary and event vocabulary; change the XML to match the source and intended measure before processing a real cohort. `dark`/`light` is configured with `<theme>` under `<preprocessing><plotting>` and controls the dashboard plus generated Plotly charts.

#### Editable operation boundaries

In both XML files, each item under `<preprocessing><relevantParameters><settings><entry name="included_phases"><array>` names a phase and defines its end event directly, instead of relying on a second boundary list. The pre-operative item's `end_event` marks **operation start**; the during-operative item's `end_event` marks **operation end/start of postoperative care**; the post-operative item's configured `end_event` marks the **post-operative-section end** used by duration calculations. If the post-operative end event is null, that boundary is unavailable and configured durations that require it show `Unavailable`. Each phase starts where the previous phase ends. For example (abbreviated):

```xml
<entry name="included_phases"><array>
  <item><object>
    <entry name="phase"><string>Pre-operative</string></entry>
    <entry name="end_event"><object>
      <entry name="event_fields"><array><item><string>epic.EVENTS.EVENT</string></item></array></entry>
      <entry name="phrases"><array><item><string>start anesthesie</string></item></array></entry>
    </object></entry>
  </object></item>
  <item><object>
    <entry name="phase"><string>During-operative</string></entry>
    <entry name="end_event"><object>
      <entry name="event_fields"><array><item><string>epic.EVENTS.EVENT</string></item></array></entry>
      <entry name="phrases"><array>
        <item><string>Afronden zorg</string></item>
        <item><string>Stop anesthesie</string></item>
        <item><string>Vertrek naar IC (of_Afdeling)</string></item>
      </array></entry>
    </object></entry>
  </object></item>
  <item><object>
    <entry name="phase"><string>Post-operative</string></entry>
    <entry name="end_event"><object>
      <entry name="event_fields"><array><item><string>epic.EVENTS.EVENT</string></item></array></entry>
      <entry name="phrases"><array>
        <item><string>admission end</string></item>
        <item><string>Getekend bezoek</string></item>
        <item><string>hospital discharge</string></item>
      </array></entry>
    </object></entry>
  </object></item>
</array></entry>
```

`event_fields` lists **CSV headers**; `phrases` lists editable **CSV cell text**. The checked-in XML also includes `event` and `events` headers and more operation-start phrases; retain or replace those that match your export. The phase-end events may use **different columns**. Values match case-insensitively, whitespace-normalized, as substrings. Each configured event needs nonempty `event_fields` and `phrases`; invalid or overlapping operation-start/end phrases are rejected. Phase items must remain in pre/during/post order. For a patient's numbered-file family, the earliest timestamped operation-start event is chosen, followed by the earliest operation-end event strictly later than it and, when configured, the earliest post-operative-end event strictly later than operation end. A timestamped value before operation start is pre-operative; `[start, operation end)` is during-operative; `[operation end, infinity)` remains post-operative for extraction and outcome review. The additional post-operative-end boundary is used by the configured hospital/ICU duration calculations; absence of that marker makes those durations unavailable, not the broader postoperative evidence. Operation-end-marker rows therefore count as postoperative. Events without a parseable timestamp cannot establish a boundary. Without operation start, pre-operative time-varying values remain missing (only configured static fallback can apply); without operation end, during values and timestamp-scoped postoperative evidence remain missing rather than being assigned to the wrong phase. Explicit `record_scope=patient` postoperative metadata can still be read from any row. The extraction report counts families lacking each boundary; it does not list their names. Multiple independent operations within one family are **not** separated into episodes; review those separately before interpreting outcomes.

#### Phase-relative time-series export

Set this shared option to `true` in `<preprocessing><relevantParameters><settings>` to write phase-relative readings:

```xml
<entry name="export_time_series"><boolean>true</boolean></entry>
```

With preprocessing enabled, `time_series.csv` contains `patient_id,phase,parameter_id,t_seconds,value`. Only selected, explicitly defined **during- or post-operative** parameters with `parameter_type=time_series_numeric` contribute numeric readings; pre-operative fields, other types, and patient-scope outcome metadata do not. A post-operative measurement's configured source fields and optional `window_hours` apply to this export too. Each family shares one pair of phase boundaries across its numbered files. During-operation `t_seconds = reading time − operation start`; post-operation `t_seconds = reading time − operation end`. Both clocks reset at their **event marker**, not at the first observed measurement; a row exactly on its phase start has `t_seconds=0`, but no synthetic zero-value row is invented when nothing was measured then. Time is elapsed **seconds**, not a wall-clock time or date; the CSV contains no absolute timestamp. Invalid/missing timestamps or nonnumeric readings are skipped; if the end marker is absent, that family contributes no series rows. Postoperative 72-hour limits, when configured, are measured from the end marker and include the endpoint. Readings are ordered by patient, then phase, parameter, and elapsed time. This file is for review/analysis, **never** a perceptron input or an outcome label. Use a fresh output directory when toggling it off: older output files are not deleted.

#### Time-series plotting

The standalone plotting command reads the preprocessed `time_series.csv`; it does not scan raw patient files. Run preprocessing with `export_time_series=true` first, then:

```powershell
py -3 -m src.cli plot-time-series
py -3 -m src.cli plot-time-series --group-by class --class epic.LAB --parameter during_creatinine --patient patient_001,patient_002
py -3 -m src.cli plot-time-series --parameter during_creatinine --patient patient_001
```

The XML pipeline can run the same plotter from `preprocessing/plotting/timeSeries`. This child has its own `run` switch, independent of the heatmap/distribution switch. For example, add these children under `<preprocessing><plotting>`:

```xml
<timeSeries>
  <run>true</run>
  <groupBy>class</groupBy>
  <parameterIds><id>during_creatinine</id></parameterIds>
  <classNames><name>epic.LAB</name></classNames>
  <patientIds><id>patient_001</id><id>patient_002</id></patientIds>
  <outputDirectory>results/time_series_plots</outputDirectory>
</timeSeries>
```

`groupBy` accepts `parameter` (one plot per parameter) or `class` (one multi-panel plot per class, with separate value axes). Empty `parameterIds`, `classNames`, and `patientIds` lists mean all eligible parameters/classes/patients; when parameter and class lists are both filled, their filters intersect. When preprocessing runs, `export_time_series` must be enabled and the plotter consumes its `time_series.csv`. When preprocessing is skipped, it reads an existing `time_series.csv` from `preprocessing/outputDirectory`. The active `pipeline.xml` enables time-series plotting with all filters empty; the starter template keeps it disabled. No shell command string is executed from XML.

By default, every configured during/post-operative `time_series_numeric` parameter and every patient in the CSV is included. `--group-by parameter` (the default) creates one interactive HTML plot per parameter, with one line per patient. `--group-by class` creates one interactive HTML plot per configured XML class, with a separate subplot/value axis for each parameter so unlike units are not mixed. Parameters outside an XML `<class>` are included in the default selection and grouped under `unclassified` in class mode.

Use `--parameter id1,id2` or repeat `--parameter` to select parameter IDs; use `--class epic.LAB` or repeat `--class` to select XML class names. When both are given, the selection is their intersection. Use `--patient patient_001,patient_002` or repeat `--patient` to plot one pseudonymous patient or compare several; omitting it includes all patients. These generated `patient_NNN` IDs are the IDs already present in `time_series.csv`, not source filenames, and can change if the analyzed input set changes. Use `--input`, `--parameter-map`, and `--output` to override the defaults. Plots are saved under `results/time_series_plots/`; Plotly charts display elapsed **hours** on the x-axis (converting the CSV's `t_seconds` by 3,600), so a 72-hour postoperative limit appears at 72 h. The y-axis uses the configured unit (or `Value`). The command logs aggregate counts only; keep both the source series and plots in approved local storage.

#### Phase-owned parameter example

```xml
<relevantParameters formatVersion="1">
  <settings><!-- Shared typed entries (see the checked-in XML). --></settings>
  <pre>
    <class name="epic.DEMOGRAPHICS">
      <timed>false</timed>
      <parameter id="age">
        <entry name="source_fields"><array><item><string>epic.DEMOGRAPHICS.age</string></item></array></entry>
        <!-- Other required workbook metadata. -->
      </parameter>
    </class>
  </pre>
  <during/>
  <post>
    <parameter id="recorded_death_days" measurementOrder="8">
      <entry name="source_fields"><array><item><string>epic.DEMOGRAPHICS.days mortality</string></item></array></entry>
      <!-- Other required workbook metadata. -->
      <entry name="measurement"><object>
        <entry name="aggregation"><string>max</string></entry>
        <entry name="record_scope"><string>patient</string></entry>
      </object></entry>
      <entry name="outcome_criteria"><object>
        <entry name="success"><null/></entry>
        <entry name="failure"><object>
          <entry name="operator"><string>gte</string></entry>
          <entry name="value"><number>0</number></entry>
        </object></entry>
      </object></entry>
    </parameter>
  </post>
</relevantParameters>
```

This abbreviated excerpt omits required metadata and other shared settings; use the checked-in XML for a complete runnable file. The optional `<timed>false</timed>` child is supported for a **pre-operative class** and applies to every parameter in that class; classes without it remain timed. In both checked-in XML files, `epic.DEMOGRAPHICS` is untimed: dashboard values appear beneath the plot, not at a fabricated operation-relative timestamp, and its columns are excluded from the operation series. Extraction still uses a known pre-operative cutoff when dated records are available to avoid postoperative leakage. If no dated pre-operative value exists, exactly one distinct non-missing value across the patient family may be used as untimed metadata; conflicting values remain missing. This setting does not make a postoperative outcome such as `days mortality` a pre-operative feature. To add a parameter, put its **definition once** in the appropriate phase, directly or inside a matching Epic class group. An empty `<parameter id="…"/>` does not select a definition elsewhere. A class uses only its nested, explicitly defined parameters; other headers in that class remain unassigned until defined. The individual outcome and measurement belong inside the post-operative parameter. Old XML using a sibling `<object>` catalog and empty selectors must be migrated. Standalone `<parameterMap>` files keep their one-object representation.

### CSV header descriptions

The starter template leaves this component **off by default**. To inspect headers without extracting patient rows or training a model, copy `config/pipeline.template.xml` to another file **in `config/`**, point `<preprocessing><inputs><path>` at the approved CSV folder/files, and set only `<preprocessing><csvHeaderDescriptions><run>` to `true`. Leave `<preprocessing><run>`, `<model><perceptron><run>`, and all other component switches `false`:

```xml
<csvHeaderDescriptions>
  <run>true</run>
  <inventoryOutput>results/csv_headers.csv</inventoryOutput>
  <summaryOutput>results/csv_header_summary.md</summaryOutput>
</csvHeaderDescriptions>
```

```powershell
py -3 -m src.cli run --config config/pipeline.headers.xml
```

`csv_headers.csv` records **source input number**, CSV filename relative to that input, family, numbered part, 1-based column position, and the **exact** header text for every CSV. The Markdown summary lists all files and counts, ordered header-schema groups, patient filename families, and each distinct header's **file** frequency (not patient frequency). At the end it adds `Pre-operative headers`, `During-operative headers`, and `Post-operative headers`, each with an exact-header table and a **class table**; a header can appear in two phase tables when separate catalog records assign it to both. The `Unassigned headers` table retains everything not selected by an exact catalog field, and the final `Epic class index` enumerates **all discovered** `epic.CLASS.item` classes, including classes with only unassigned headers. Phase assignment is from the XML, **not inferred from a class name or from patient rows**. Non-Epic fields may appear in a phase header table but have no Epic class.

The family convention matches preprocessing: `ref_0002.csv` and `ref_0002_1.csv` are one patient, while `ref_0003.csv` is another; a zero-padded final identifier of three or more digits remains part of the base name. `case_1.csv` and `case_2.csv` share the `case` family even without `case.csv`. Families never span directories, but can include separately configured input files in the same directory. Source input numbers remain in the per-file inventory. Identical headers in different files are still recorded per file; duplicate columns in a single file retain both positions. Quoted commas and tab-delimited headers are supported. Source directories are scanned recursively; the configured `outputDirectory` is excluded from input discovery.

Only the first CSV record is decoded for each file, **not** patient rows or measurements. File names themselves may be patient identifiers: treat both reports as sensitive, store them only in approved local storage, and never commit or post them. Logs include aggregate counts, not filenames. The inventory preserves header text exactly; open it in a text/CSV-aware viewer rather than a spreadsheet that could interpret untrusted header strings as formulas. This inventory does **not** modify the workbook or the embedded `relevantParameters` catalog; it provides evidence to review their field coverage separately.

### `model`

`activeModel` selects a model name registered to a class in `src/model_config.py`. Each model class parses its own child nodes and can compose reusable nodes such as `TrainingValidationConfig`, `NovelDataConfig`, and `PlotConfig`. **Perceptron is the only model implemented now.** To add another runnable model, add and register its class, add its name and XML type/section to the XSD, update both XML files, add its runner in `pipeline_runner.py`, and test its configuration and execution behavior.

### `model/perceptron`

- `run` enables or skips perceptron execution when it is the active model.
- `mode` is either `train-validation` or `novel-data`.
- `trainingValidation` sets input vectors/outcomes when preprocessing is skipped, the labeled training CSV, the prediction-only validation CSV, model output, train fraction, epochs, seed, and the XML `omit/id` list. The default config omits `operation_diagnosis_proxy`; change that list to change exclusions. The direct CLI `--omit` option adds further exclusions. If preprocessing runs, training consumes its vectors and outcomes automatically.
- `novelData` selects feature-only vectors, a previously saved model, and the output predictions CSV. If preprocessing runs, the freshly generated vectors are used. Novel prediction CSVs contain `prediction`, never a postoperative `target`.
- `plotting/run` optionally writes an aggregate, interactive Plotly confusion-matrix HTML file to `plotting/output`; it is only valid in `train-validation` mode. Keep the adjacent `plotly.min.js` bundle with it for offline viewing.

### `model/perceptron/mode` examples

For training and holdout validation:

```xml
<mode>train-validation</mode>
```

For applying a saved model to novel feature vectors:

```xml
<mode>novel-data</mode>
```

Set the corresponding paths under `trainingValidation` or `novelData`. The model uses only configured pre-operative features; postoperative data remains outcome evidence, not model input.

## Editing and validating

Edit the master file with an XML editor such as XML Notepad; installation details are in the [XML parameter-map guide](parameter-map-xml.md). The master XML points to `pipeline.xsd` through `xsi:noNamespaceSchemaLocation`. From the repository root, validate the master file and template with `xmllint` when available:

```text
xmllint --noout --schema config/pipeline.xsd config/pipeline.xml config/pipeline.template.xml
```

The XSD checks document structure, values, and allowed modes. The Python loader separately checks runtime requirements such as supported phase names, positive training settings, and whether model plotting is compatible with the selected mode. Schema validation alone does not guarantee the input files exist or that there are enough labeled cases to train.

Paths in the XML are relative to the project root (the parent of `config/`); keep alternate run-control XML files directly in `config/` so relative paths resolve consistently. The direct `analyze`, `distributions`, and `train-perceptron` commands remain available, but `run` is the single XML-driven workflow.

## Extending the schema and pipeline

Edit only `config/pipeline.xsd`. It owns both global XML roots (`pipelineConfiguration` and `parameterMap`) and the shared typed value and phase-definition types. Every element declaration and named type has an `xs:annotation/xs:documentation` description; add the same kind of short, user-facing explanation for every new element, attribute, or type. For example:

```xml
<xs:element name="run" type="xs:boolean">
  <xs:annotation>
    <xs:documentation>Whether this component is executed.</xs:documentation>
  </xs:annotation>
</xs:element>
```

Use `xs:sequence` for ordered children. Its order is the XML order and the intended execution order. Use `xs:complexType` for a component's child settings, and `xs:simpleType` restrictions for allowed values such as model names and execution modes.

### Add a setting or component

- **Root-level shared setting:** add the element to the `pipelineConfiguration` `xs:sequence` in `config/pipeline.xsd`, add it directly under `<pipelineConfiguration>` in both XML files, document it with `xs:documentation`, parse it in `load_pipeline_config`, and pass it to the logger or stage that consumes it. Shared settings are not a processing stage by themselves.
- **Preprocessing component:** add its element at the intended position in `PreprocessingType`, define/document its type, and add the same node to both XML files. Parse and validate its values in `pipeline_config.py`; implement the component and call it in `pipeline_runner.py` before the model stage when that is its declared position. Give executable components an explicit `run` switch and test enabled and disabled behavior using synthetic inputs.
- **Model:** define a `ModelConfig` subclass in `src/model_config.py` and register it with its XML model name. Parse the model's own children there, reusing node classes such as `TrainingValidationConfig`, `NovelDataConfig`, or `PlotConfig` where the XML shape matches. Add its name to `ModelName`, define and document its XSD type/section, and add it to both XML files. Register its execution handler in `pipeline_runner.py`; test class parsing, required/invalid children, modes, and input/output contracts. `perceptron` is the only implemented model now, so registering a parser alone does not make a model runnable.
- **New pipeline node:** add the node to the root `pipelineConfiguration` sequence at its execution position, create/document its type, update both XML files, parse it, and add its execution handler at the same position in `pipeline_runner.py`. Add a synthetic end-to-end test that proves the ordering and `run` behavior.

After changes, validate the active file and template with the command above, run `py -3 -m unittest discover -s tests -v`, and update this guide if the node's purpose or ordering changes. The same XSD can validate a standalone catalog explicitly with `xmllint --noout --schema config/pipeline.xsd path/to/catalog.xml`; standalone maps need not carry a second schema reference.
