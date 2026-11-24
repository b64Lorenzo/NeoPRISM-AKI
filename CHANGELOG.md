# Changelog

Project releases use `vX.Y.Z` tags. See [the release process](docs/release-process.md).

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
