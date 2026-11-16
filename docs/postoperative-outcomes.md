# Post-operative parameters and binary outcome rules

Post-operative values are evidence for outcome labels, **not perceptron inputs**. A matched per-parameter failure criterion yields `-1`; otherwise, a matched success criterion yields `+1`. If both kinds of evidence are present, failure takes precedence. `NA` is retained only for missing, unmatched, or not-yet-defined evidence; it is not a third class and is excluded from training and evaluation.

Individual rules live in the typed `outcome_criteria` entry of their **own post-operative `<parameter>`** inside `config/pipeline.xml`; there is no separate `<outcome id="...">` or shared definition to edit. For example (other required metadata and measurement entries omitted):

```xml
<postOperative>
  <parameter id="mortality_30d">
    <entry name="outcome_criteria"><object>
      <entry name="success"><object>
        <entry name="operator"><string>equals</string></entry>
        <entry name="value"><number>0</number></entry>
      </object></entry>
      <entry name="failure"><object>
        <entry name="operator"><string>equals</string></entry>
        <entry name="value"><number>1</number></entry>
      </object></entry>
      <entry name="status"><string>provisional</string></entry>
    </object></entry>
  </parameter>
</postOperative>
```

Supported operators are `equals`, `in`, `lt`, `lte`, `gt`, and `gte`. Without a clinically defined criterion, put `<entry name="success"><null/></entry>` and `<entry name="failure"><null/></entry>` inside `outcome_criteria` and set `status` to `pending_clinical_definition`. Omitting `outcome_criteria` also means no criterion; no medical threshold is inferred from the parameter's range. See the [XML parameter-map guide](parameter-map-xml.md) for the complete format.

A post-operative class group can opt into shared failure, for example `<class name="epic.FLUIDS_OUT" failure="nonempty"><parameter id="postop_urine_72h">…definition…</parameter></class>`. It examines only explicitly defined post-operative fields in that class. A nonblank value not in the configured missing-value tokens—even `0`, `false`, or a negative number—makes the patient a failure; extra CSV columns in the class do not. Normal timestamp constraints measured from the XML-configured **operation end** and parameter `window_hours` constraints apply; `record_scope=patient` explicitly permits patient-level metadata from any row. Class failure takes precedence over individual success. **No class-wide failure is enabled in the checked-in XML.** Do not enable it for broadly populated classes without clinical validation that *any value* means failure.

## Post-operative parameters

| Parameter | Source / derivation | Success | Failure | Rule status / note |
|---|---|---|---|---|
| `postop_creatinine_72h` | Maximum creatinine in configured 72-hour window | Not defined | Not defined | Pending; an isolated absolute value needs a clinically validated baseline-relative rule. |
| `postop_urine_72h` | Maximum recorded urine value | Not defined | Not defined | Pending; interval and weight normalization are unavailable. |
| `ventilation_duration` | `q_umcg_aki_marco.adm_ventilationdays` | Not defined | Not defined | Pending; no cutoff supplied. |
| `icu_los` | Candidate fields `q_umcg_aki_marco.adm_ic_admdate` and `q_umcg_aki_marco.adm_ic_disdate`; sum of deduplicated/merged ICU-interval overlap with the configured operation-start–post-operative-end window | Not defined | Not defined | Configured duration for the dashboard summary; ISO 8601 and `Europe/Amsterdam` are unverified assumptions. Confirm fields, datetime representation, timezone, and whether the window matches the intended ICU measure before cohort use. |
| `hospital_los` | Configured operation-end boundary to earliest matching post-operative-end event (`admission end`, `Getekend bezoek`, or a configured discharge phrase) | Not defined | Not defined | Dashboard interval proxy in days; a signed visit alone is not verified discharge evidence. Confirm event meaning against source exports before interpreting as hospital stay. |
| `mortality_30d` | `q_umcg_aki_marco.adm_deceased` | Recorded flag `0` | Recorded flag `1` | Provisional; the field's mortality horizon is unverified and may not mean 30-day mortality. |
| `recorded_death_days` | Exact header `epic.DEMOGRAPHICS.days mortality` | Not defined (blank/NA does not prove survival) | Any finite number ≥ 0, including `0` | User-defined, provisional; recorded death with unknown follow-up horizon, **not** a 30-day endpoint. User-configured source field. |
| `postop_ecmo` | No confirmed source field | Not defined | Not defined | Pending / unavailable. |
| `prism_ii` | `q_umcg_aki_marco.adm_PRISMII` | Not defined | Not defined | Pending; no cutoff supplied. |
| `pim` | `q_umcg_aki_marco.adm_PIM` | Not defined | Not defined | Pending; timing, definition, and cutoff need confirmation. |

Direct postoperative categorical values in the configured `operation result` fields are separately mapped in `patient_outcome_rules.categorical_rules`: `optimal`/`mid` values are success and `severe`/death values are failure. Unknown category values do not classify.

`recorded_death_days` is a **patient-level outcome field**, not an input feature: `record_scope=patient` reads it from any row, including a row timestamped before surgery, because the demographic value may be repeated as patient metadata rather than recorded at the time of death. The configured numeric `max` aggregation ensures that any nonnegative numeric value among the rows is failure evidence; blank/NA, non-numeric, and negative values alone remain unclassified. Failure takes precedence over other success evidence. No maximum day or 30-day threshold is inferred. Verify that the recorded death follows the relevant operation and confirm units/follow-up with the data owner before interpreting it as a clinical endpoint.

## Extracted files

- `patient_vectors.csv`: patient ID and pre-operative features only.
- `patient_outcomes.csv`: patient ID, `target`, `target_status`, and `label_basis`; no score column.
- `postoperative_measurements.csv`: aggregated timestamped measurements from operation end onward, kept separate from model inputs. Without a matched end, timestamp-scoped evidence and direct categories remain unavailable; explicit `record_scope=patient` metadata may still contribute. Configure CSV event headers and end phrases in `duringOperative/endEvent` in [the XML guide](xml-pipeline-configuration.md#editable-operation-boundaries).
- `time_series.csv`: optional unaggregated, numeric during/post-operative readings when XML `exportTimeSeries` is enabled. Postoperative `t_seconds=0` is the matched end event, not the first recorded result; measurement `window_hours` applies. No dates or absolute timestamps are written. See [phase-relative export](xml-pipeline-configuration.md#phase-relative-time-series-export).

Raw laboratory values are retained as measurements without automatically inferring an outcome. Confirm all endpoint definitions, units, timestamps, source-field meanings, and the mortality horizon with the data owner and clinical collaborators before research use. These prototype rules are not clinically validated decision-support rules.
