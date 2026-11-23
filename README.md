# NeoPRISM-AKI

[![Latest release](https://img.shields.io/github/v/release/b64Lorenzo/pattern-recognition-course)](https://github.com/b64Lorenzo/pattern-recognition-course/releases/latest)

![NeoPRISM-AKI — perioperative research framework](assets/branding/banner.png)

NeoPRISM-AKI is a research-only Python framework for retrospective perioperative data. It preserves configurable parameter extraction, phase-aware missingness views, patient-level model inputs, and a **binary Rosenblatt perceptron** baseline.

**The project does not infer clinical AKI outcomes from missing measurements.** A clinically adjudicated label CSV is required for training. Unknown labels remain unknown and are excluded from training, never converted to No AKI. The model produces **hard class labels and uncalibrated linear scores**, not AKI risk probabilities. It is not a clinical decision-support system.

![NeoPRISM-AKI preprocessing and model workflow](assets/branding/workflow.png)

## Quick start

From the project root with Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
uv run python -m src.cli --help
uv run python -m pytest -q tests src/models/rosenblatt_perceptron/tests src/tests
```

1. Keep your existing `config/parameters.ini` and `config/plotting.ini`; merge the supplied `*.additions.ini` snippets without duplicate section headers.
2. Run parameter extraction to populate `results/parameters/{preoperative,intraoperative,postoperative}/ref_*.csv`.
3. Copy `config/inputsModel.example.ini` to `config/inputsModel.ini`, and edit the exact `phase/parameter_id` features to match your parameter catalog.
4. Create a clinically reviewed label CSV with `patient_id,aki_label` as documented in [model inputs and labels](docs/model-inputs-and-labels.md). Missing outcomes remain unknown, never negative.
5. Run `python -m src.cli inputs-model --config config/inputsModel.ini`.
6. Run `python -m src.cli rosenblatt-train --input results/inputsModel/patients_labeled.csv --output results/models/rosenblatt_perceptron`.
7. For new patients, run `python -m src.cli inputs-model --config config/inputsModel.ini --unlabeled` on the new extraction directory, then `python -m src.cli rosenblatt-predict --model results/models/rosenblatt_perceptron/model.json --input results/inputsModel/patients_unlabeled.csv --output results/models/rosenblatt_perceptron/novel_predictions.csv`.

See [model-input documentation](docs/model-inputs-and-labels.md), [model documentation](docs/rosenblatt-perceptron.md), [plotting documentation](docs/plotting-and-patient-dashboards.md), [release process](docs/release-process.md), and [project structure](docs/code-structure.md).

## Research safeguards

- All features are explicitly selected by phase; postoperative observations must not be used as predictors for postoperative AKI.
- For preoperative prediction select only preoperative features. End-of-surgery predictors may include intraoperative summaries available by surgery end. `operation_duration` is **not** available before operation end.
- For sequential snapshots, specify `cutoff_minutes` and use only observations genuinely available by the cutoff; extraction timestamps alone may not capture laboratory reporting delay.
- Patients are split before feature preprocessing. Numeric imputation/standardisation and categorical vocabularies are learned from training patients only.
- The split is patient-disjoint and stratified when each class has at least two patients. External validation, confidence intervals, calibration, temporal drift and outcome adjudication are still needed.
- CSVs and `split.json` contain patient identifiers: store them in approved locations, not a public Git repository. Model JSON contains weights and preprocessing statistics, but no patient identifiers.

## Tests

```bash
python -m pytest -q tests src/models/rosenblatt_perceptron/tests src/tests
```
