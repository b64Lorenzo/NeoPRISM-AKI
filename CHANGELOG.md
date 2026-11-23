# Changelog

Project releases use `vX.Y.Z` tags. See [the release process](docs/release-process.md).

## [0.1.0] — 2026-11-23

First tagged research-framework release.

### Added

- Configurable, phase-aware perioperative parameter extraction and missingness plots.
- Patient-level model-input workflow and a binary Rosenblatt perceptron baseline.
- NeoPRISM-AKI identity assets, workflow illustration, and a documented release convention.

### Safety and scope

- Training requires a clinically reviewed label CSV; missing observations are not converted into negative labels.
- This release is research software only. It is not clinically validated and is not for clinical decision-making.
