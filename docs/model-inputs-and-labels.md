# Patient-level model inputs and AKI outcome labels

## Location and contract

`src/preprocessing/inputsModel/` contains `config.py`, `labels.py`, `builder.py` and `run.py`. The stage reads the **existing extracted phase CSVs** and creates one row per `patient_id` (or surgical episode ID if that is how the extractor identifies episodes). A selected feature is named `phase/parameter_id`, e.g. `preoperative/bmi` or `intraoperative/heart_rate`. The parameter must exist in the extraction catalog.

The output is `results/inputsModel/patients_labeled.csv` with `patient_id`, configured feature columns, and `aki_label`, `aki_stage`, `label_status`, `label_reason`. An unlabeled run writes `patients_unlabeled.csv` with the same feature columns and no target. `summary.json` records patient counts, missing feature counts and label counts.

## Configuration

Copy `config/inputsModel.example.ini` to `config/inputsModel.ini`. `[paths]` identifies the extracted CSV directory, `config/parameters.ini`, the adjudicated label CSV and output directory. `[features] parameter_ids` lists exact `phase/parameter_id` columns. `[features] aggregation` supports `patient_mean`, `patient_median`, `patient_min`, `patient_max`, `patient_first`, and `patient_last`. Categorical fields use the first or last nonempty observation, rather than a numeric mean. `cutoff_minutes` is optional; when set, intraoperative observations later than that minute are excluded, and postoperative predictors are prohibited.

## Label protocol

`data/labels/aki_labels.csv` has columns `patient_id,aki_label`. In binary mode: `-1` means **No AKI**; `+1` or `1` means **AKI**; an empty field means **unknown/not adjudicated**. Alternatively `kind = kdigo_stage` interprets clinically adjudicated `0,1,2,3` as binary `-1,+1,+1,+1` respectively; the original ordinal stage is not retained in the binary model matrix, so retain the clinical source separately.

Labels can be externally adjudicated or provisionally derived from extracted creatinine, depending on `[labels] source`. See [AKI labeling protocol](aki-labeling-protocol.md). For the intended postoperative 72-hour endpoint, a clinician-approved KDIGO/neonatal protocol must establish baseline creatinine, urine-output criteria, assessment window, data sufficiency and stage. Missing postoperative measurements do not imply stage 0. A label is per patient/episode, not per feature. If repeated operations occur for the same individual, use an episode-aware ID and split/group by the underlying patient for unbiased evaluation; the current split enforces unique row IDs but does not automatically group distinct episodes from the same person.

## Feature availability and leakage

Preoperative predictions should use only preoperative variables. End-of-operation predictions may include intraoperative summaries, including duration if both boundaries are known. Postoperative creatinine used to construct the AKI outcome must never be included as a predictor. The `cutoff_minutes` setting is a basic temporal filter; it does not reconstruct reporting/availability timestamps. For prospective sequential studies, enforce acquisition and availability time separately.

## Data quality

The builder does not silently fill missing features. Empty feature cells remain empty in the matrix. During training, median imputation for numeric fields and a missing-category indicator are learned from the **training split only**. Unseen categorical values encode as no known category; do not treat this as clinically equivalent to a known reference category.
