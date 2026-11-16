# Pattern Recognition

Course project for a configurable patient-CSV workflow: extract pre-operative features, derive postoperative labels, and train a perceptron baseline.

## Quick start

Requires Python 3.10 or newer.

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m unittest discover -s tests -v
py -3 -m src.cli --help
```

## Run the pipeline

Before running the active configuration, review `config/pipeline.xml`: its input path or enabled stages may refer to an external cohort. The disabled `config/pipeline.template.xml` is the starter. Keep cohort inputs, patient-level outputs, and logs in approved storage; default output folders are ignored by Git, but custom folders may not be.

```powershell
py -3 -m src.cli run --config config/pipeline.xml
```

For direct workflows, use `analyze` on an explicitly selected cohort folder and train from its outputs:

```powershell
py -3 -m src.cli analyze "C:\path\to\patient-csvs" --output "C:\path\to\results"
py -3 -m src.cli train-perceptron --help
py -3 -m src.cli distributions "C:\path\to\patient-csvs" --output "C:\path\to\results" --phase all --bins 20
py -3 -m src.cli plot-time-series --input results/time_series.csv
```

## Data and outputs

Same-directory numbered CSV parts are grouped into one patient family. `patient_vectors.csv` contains pre-operative inputs only; `patient_outcomes.csv` stores postoperative-derived labels separately. Training joins by `patient_id`. Labels are `-1` (failure), `+1` (success), or `NA` (unclassified); failure evidence takes precedence, and `NA` is excluded from training. Validation exports predictions without the true target. The perceptron is a hard classifier, not a calibrated probability estimator.

Analysis also writes `postoperative_measurements.csv` for review. `duringoperative_measurements.csv` and phase-relative `time_series.csv` are optional. The checked-in `during_creatinine` mapping is illustrative; verify source availability, timing, and units. Distribution and missingness plots have independent phase, parameter-ID, and mapped-class filters in XML; these filters affect plots only, not model inputs. Heatmaps, a patient timeline, distributions, and time-series plots are configurable; see the guides below. Existing output files are not cleaned up between runs—use a fresh directory when needed. Logs are written to `logs/<UTC-run-id>/run.log` and mirrored to stderr.

All automated fixtures are synthetic; they do not establish clinical validity. Confirm field meanings, units, event boundaries, and outcome definitions with the data owner before cohort use.

## Project guides

| Topic | Guide |
|---|---|
| Current CLI extraction and outputs | [CSV extraction](docs/data-extraction.md) |
| Patient grouping, vectors, and artifacts | [Patient-vector workflow](docs/patient-vectors.md) |
| Feature definitions and BMI | [Pre-operative parameters](docs/preoperative-parameters.md) |
| Outcome evidence and limitations | [Post-operative outcomes](docs/postoperative-outcomes.md) |
| Training and validation | [Perceptron guide](docs/perceptron-training.md) |
| Edit and validate the parameter catalog | [XML parameter map](docs/parameter-map-xml.md) |
| Pipeline stages, settings, and migration | [XML pipeline configuration](docs/xml-pipeline-configuration.md) |
| Packages and extension points | [Code structure](docs/code-structure.md) |

On Windows, XML Notepad is an optional XML editor; installation and validation steps are in the [parameter-map guide](docs/parameter-map-xml.md). Original project code and documentation are MIT-licensed; course materials, datasets, and third-party content retain their own terms ([notices](THIRD_PARTY_NOTICES.md)).

## AI use notice

Code autocompletion is used. AI may assist with writing tests and documentation. Authors are responsible for reviewing and validating all submitted work.