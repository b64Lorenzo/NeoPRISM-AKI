# Configurable AKI labeling — neonatal KDIGO research protocol

## Source and interpretation

Zappitelli M et al., *Developing a neonatal acute kidney injury research definition: a report from the NIDDK neonatal AKI workshop*, Pediatric Research 82 (2017), 569–573, doi:10.1038/pr.2017.136, Table 1, p. 571. This paper **discusses proposed research modifications**, not a fully validated postoperative 72-hour algorithm. The example 72-hour window is a project-specific choice, not a time window prescribed by this paper.

The serum-creatinine thresholds implemented in `clinical_labels.py` are:

| Stage | Research serum-creatinine criterion |
|---|---|
| 0 | No threshold met in observed values (creatinine-only provisional negative) |
| 1 | At least 1.5 times baseline, or at least 0.3 mg/dL increase within 48h between *timed postoperative samples* |
| 2 | At least 2 times baseline |
| 3 | At least 3 times baseline, or at least 2.5 mg/dL for `neonatal_2017` profile |

The `kdigo_2012` profile uses 4.0 mg/dL instead of 2.5 mg/dL for the absolute stage-3 cutoff. The standard 7-day ratio and 48-hour absolute-change criteria are evaluated only within the configured observation window. The algorithm does **not** infer the 48-hour change from an undated preoperative sample.

**Not implemented as automatic criteria:** urine-output stages, anuria, renal replacement therapy, gestational-age-dependent adjustments, maternal creatinine considerations, and reliable assessment of observation completeness. These require validated longitudinal urine volume/weight and coverage data or external adjudication. Use `source = external` with `kind = kdigo_stage` to incorporate these criteria, or `external_override` if you explicitly accept creatinine-only fallback. Do not interpret provisional negatives as verified No AKI.

## Strategies

- `source = creatinine`: automatically derive a provisional stage from extracted `kreatinine` values. Does not need `data/labels/aki_labels.csv`.
- `source = external`: read a reviewed CSV with `patient_id,aki_label` (`kind=binary`) or `patient_id,aki_stage` (`kind=kdigo_stage`).
- `source = external_override`: use a nonempty reviewed label if present; otherwise use the creatinine-derived stage.
- `source = consensus`: accept a label only if the external label and creatinine-derived binary result agree. This is a *selection strategy*, not an independent clinical consensus panel.

### Example INI

```ini
[paths]
input_directory = results/parameters
parameter_config = config/parameters.ini
labels_csv = data/labels/aki_labels.csv
output_directory = results/inputsModel

[features]
parameter_ids = preoperative/age, preoperative/sex, preoperative/bmi
aggregation = patient_mean
cutoff_minutes =

[labels]
source = creatinine
kind = binary
column = aki_label

[clinical_aki]
creatinine_parameter_id = kreatinine
creatinine_unit = umol/L
baseline_strategy = preoperative_last
window_hours = 72
threshold_profile = neonatal_2017
include_intraoperative_baseline = false
```

Use `preoperative_min` for the minimum timed preoperative creatinine; `preoperative_last` for the latest; `previous_lowest` uses the lowest earlier postoperative measurement (and preoperative baseline) for relative changes. All require a valid preoperative baseline in this implementation. **Baseline strategy should be chosen prospectively and justified by neonatal age and physiology.**

The extraction `results/parameters/summary.csv` must include `patient_id,operation_end_minutes`. Postoperative creatinine samples are filtered to `[operation_end_minutes, operation_end_minutes + window_hours*60]` on the extraction's operation-start-relative clock. Without an operation-end boundary, a baseline, or postoperative samples, the label is **unknown**, never automatically -1.

## Output

`results/inputsModel/patients_labeled.csv` contains the selected pre/intraoperative features and:

- `aki_label`: `-1`, `1`, or blank.
- `aki_stage`: `0`, `1`, `2`, `3`, or blank.
- `label_status`: `determined` or `unknown`.
- `label_reason`: e.g. `creatinine_only`, `no_postoperative_creatinine_in_window`, `external_adjudicated`, or `missing_or_discordant_labels`.

The Rosenblatt trainer excludes these metadata columns from predictors; only configured phase/parameter feature columns are used. **Do not include postoperative creatinine as a predictor of postoperative creatinine-defined AKI.**

## Commands

```powershell
uv run python -m src.cli parameters --help
uv run python -m src.cli inputs-model --config config/inputsModel.example.ini
uv run python -m src.cli rosenblatt-train --input results/inputsModel/patients_labeled.csv --output results/models/rosenblatt_perceptron
uv run python -m pytest -q tests src/models/rosenblatt_perceptron/tests src/tests
```

If your extracted dataset lacks `kreatinine`, add the appropriate source field to `config/parameters.ini` and rerun extraction. If the parameter is named differently, set `creatinine_parameter_id` accordingly. If the source units differ, use a supported unit (`umol/L` or `mg/dL`) and verify the conversion against the raw dataset. **The module does not inspect per-row units:** the configuration must match the extracted values.

### Clinical validation checklist

Verify age/population suitability, unit correctness, timing and sample coverage, baseline selection, neonatal stage-3 cutoff, RRT/urine-output completeness, repeat operations, and the distinction between an observed negative and a fully adjudicated negative. Use external clinician-reviewed labels for primary clinical evaluation.
