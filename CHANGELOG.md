# Changelog

Project releases use `vX.Y.Z` tags. See [the release process](docs/release-process.md).

## [0.3.0] — 2026-11-26

### Changed

- Match the Rosenblatt numerical trainer to `nnci-course-python` commit `be038b430b599df713ee9de01ef54290306d1671`: zero initialization, fixed example order, per-coordinate `y*x/N` updates, per-pattern embedding strengths, and strict positive-margin convergence independent of the update threshold.
- Default to homogeneous training; expose `--clamped` / XML `clamped` for explicit constant-input augmentation using denominator N+1. Remove custom Rosenblatt learning-rate overrides and accept any finite update threshold, matching the reference.
- Keep schema-version-1 saved-model inference compatible. Retraining intentionally differs from the earlier always-biased, shuffled adaptation; document migration and distinguish update-threshold violations from convergence diagnostics.
- Add exact reference-oracle comparisons and synthetic CSV/CLI/XML integration coverage. No patient data processed by verification.

## [0.2.0] — 2026-11-24

Adds additional model families and an optional XML-configured pipeline.

### Added

- Multinomial logistic-regression and multilayer-perceptron training/prediction workflows, with explicit target selection and shared evaluation reports.
- XML pipeline configuration, schema validation, deterministic planning, and execution for preprocessing and model stages.
- Support for using the explicit `aki_stage` target with the new classifiers.

### Changed

- Share model-data and evaluation helpers across classifiers and streamline the Rosenblatt perceptron implementation.
- Add synthetic regression coverage for model, CLI, numerical-failure, and XML-pipeline behavior.

### Safety and scope

- No patient/cohort records or local clinical mappings are included in this release; automated tests use synthetic fixtures.
- Research software only; this release adds no clinical validation, calibration, or decision-support claims.

## [0.1.0] — 2026-11-23

First tagged research-framework release.

### Added

- Configurable, phase-aware perioperative parameter extraction and missingness plots.
- Patient-level model-input workflow and a binary Rosenblatt perceptron baseline.
- NeoPRISM-AKI identity assets, workflow illustration, and a documented release convention.

### Safety and scope

- Training requires a clinically reviewed label CSV; missing observations are not converted into negative labels.
- This release is research software only. It is not clinically validated and is not for clinical decision-making.
