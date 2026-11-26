# Classifier model configuration

[`config/models.example.ini`](../config/models.example.ini) contains training and
local Ray resource settings for the multilayer perceptron, hard multi-output
perceptron and logistic comparator. It contains no cohort paths or clinical data.
Reading it validates settings only: no worker starts and no directory is created.

## Train, select a profile, predict

```bash
uv run python -m src.cli mlp-train --config config/models.example.ini --input path/to/matrix.csv --output path/to/mlp
uv run python -m src.cli multioutput-train --config config/models.example.ini --input path/to/matrix.csv --output path/to/multioutput
uv run python -m src.cli logistic-train --config config/models.example.ini --input path/to/matrix.csv --output path/to/logistic
```

Replace the placeholder paths before training. With `--config`, the INI is
**authoritative for architecture and training**. Explicit architecture/training
CLI flags are errors even if their values equal the INI values; parser defaults
must not silently override settings. Without `--config`, existing flag-only
commands keep their defaults and remain supported, **still using real Ray**.
The pure numerical `.fit()` APIs remain NumPy-only for use as kernels; the
patient-level `classification.fit`, CLI and XML orchestrate Ray. Data/experiment settings,
including input/output paths, target, ordered features, split fraction and seed,
stay in the CLI or XML. The seed continues to control splitting and initialization.

Each model section requires `model_type`. Unless `--profile NAME` is supplied,
the selected section name equals the command's canonical type:

| Command | Default profile and required `model_type` |
| --- | --- |
| `mlp-train` | `multilayer_perceptron` |
| `multioutput-train` | `multi_output_perceptron` |
| `logistic-train` | `logistic_regression` |

Additional section names are permitted, for example:

```ini
[compact_mlp]
model_type = multilayer_perceptron
hidden_layers = 4
hidden_activations = tanh
init_methods = xavier,xavier
```

Select that section using `mlp-train --config path/to/models.ini --profile compact_mlp`.
`--profile` requires `--config`; a profile whose declared type differs from the
command's type is an error. **Every declared profile is validated**, not just the
selected one, so an invalid unused section also prevents loading.

Prediction is artifact-only: use the saved model and frozen transformer, not a
training INI or new architecture overrides. Prediction does not require Ray and
does not refit preprocessing.

### XML references

Within an existing model's `<options>`:

```xml
<option name="config" value="models.example.ini"/>
<option name="profile" value="compact_mlp"/>
```

The `config` reference is relative to the XML file; the optional `profile` selects
a section. The shown profile must exist in the referenced INI. Do not combine an
INI reference with inline architecture/training options. XML retains ownership
of that model's input/output paths, target, features and split. Disabled model
entries do not open their referenced INIs or start Ray. XML structural/reference
validation and validation of an enabled model's INI content are separate steps;
a disabled reference does not prove the INI exists or is valid. Enabled model
INIs are preflighted before any preprocessing handler runs. Matrix files are
still opened only when the model executes, after preprocessing.

## Settings and defaults

`[ray]` is optional; its omitted keys use the defaults below. Every configured
training profile uses **actual Ray with a local worker**, not Ray local-mode or a
silent in-process fallback. There are no backend, remote-cluster address or GPU
keys. Each fit runs as **one worker task**; this is not Ray Train/DDP or a
data-parallel optimizer. Increasing CPU settings does not partition the dataset
or run multiple experiments. CPU counts are logical scheduling resources, not
hard physical thread limits. A pre-initialized Ray context is refused rather
than reusing or shutting down another job's runtime. Owned runtimes are shut down
on success/failure; task timeouts cancel the task. There is no automatic retry or
fallback. Telemetry defaults off unless `RAY_USAGE_STATS_ENABLED` is already set.
Saved training metadata records Ray version, worker/driver PIDs and worker ID.

| `[ray]` key | Default | Validation / meaning |
| --- | --- | --- |
| `num_cpus` | `1` | Positive integer local CPU resource count |
| `cpus_per_task` | `1.0` | Positive finite CPU reservation, no larger than `num_cpus`; fractions allowed |
| `timeout_seconds` | `300.0` | Positive finite maximum task wait, in seconds |
| `temp_directory` | empty | Runtime uses `neoprism-ray` beneath Python's temporary directory; otherwise the configured resolved path |

A relative `temp_directory` resolves against the **resolved INI's directory**, not
cwd or the referring XML's directory. Absolute paths stay absolute. No `~`,
environment-variable or INI interpolation expansion is performed. The parser
resolves the path but neither checks for directory existence nor creates it.
A relative CLI `--config` path itself is relative to cwd.

**Windows:** Ray's Windows support is beta; see the official
[installation/platform notes](https://docs.ray.io/en/latest/ray-overview/installation.html).
Keep a configured Ray temporary directory short, for example `C:/raytmp`. Ray
creates a dashboard port filename of roughly 80 characters inside its session
directory; a deeply nested per-test/output path can exceed `MAX_PATH` even when
the root path alone fits. The execution adapter rejects excessively long native
Windows runtime paths before startup, with a request to choose a shorter path;
the INI parser itself only resolves paths. It does not create the example
directory. A successful parse is not proof that Ray can start on the host. Startup/task failures must remain visible, never trigger
a different execution backend.

### Gradient-trained models

Both MLP and logistic profiles accept:

| Key | Default | Validation |
| --- | --- | --- |
| `epochs` | `1000` | Positive integer maximum epoch count |
| `learning_rate` | `0.05` | Positive finite number |
| `batch_size` | `32` | Positive integer; the final partial batch is allowed |
| `tolerance` | `0.0` | Nonnegative finite training-loss improvement threshold; zero disables early stopping |
| `output_activation` | `softmax` | Must remain `softmax`, including binary classification |
| `loss` | `cross_entropy` | Must remain `cross_entropy` |

Training history is not validation performance. Output width is the number of
observed training classes; input width is inferred after fitting preprocessing.
The INI cannot hardcode either width. Observed class order is saved with the model.

### Multilayer perceptron

- `hidden_layers`: comma-separated positive integer widths, default `16`.
  An explicitly empty value means **no hidden layers**, not the default width.
- `hidden_activations`: one name per hidden layer, in forward order. Choices are
  `relu`, `leaky_relu`, `sigmoid`, `tanh`, `linear`. Omission supplies ReLU for every
  configured hidden layer; an explicit list is never broadcast. Empty is valid
  only when there are no hidden layers. Softmax cannot be a hidden activation.
- `init_methods`: optional comma-separated names, each `he`, `xavier` or `zero`.
  If supplied, it covers **all hidden layers plus the output layer**. Thus widths
  `16,8` and activations `relu,tanh` can use `he,xavier,xavier`. With no hidden
  layer, there must still be one output initializer. An explicitly empty list is
  invalid. Omission yields `None`, asking the numerical model to use automatic He
  initialization for ReLU/leaky ReLU and Xavier otherwise. Zero is supported but
  can preserve unwanted hidden-unit symmetry.

`l2` is not an MLP INI option. Other numerical Python API tasks, such as a single
sigmoid/BCE output or regression/MSE, are separate from this classification-only
INI contract and are not enabled by changing its output/loss keys.

### Hard multi-output perceptron

`multi_output_perceptron` is a genuine hard multiclass mistake-driven classifier:
linear class scores, argmax decisions and online updates on mistakes. Its only
training keys are positive integer `epochs=1000` and positive finite
`learning_rate=0.05`. `output_activation` must be `linear`; `loss` must be
`mistake_rate`. It rejects hidden layers, activations/initializers, minibatches,
tolerance and L2. Its scores are not class probabilities.

This is **not** zero-hidden softmax/logistic regression. Rosenblatt's separate
binary course algorithm and configuration remain unchanged.

### Logistic comparator

`logistic_regression` uses full K-class softmax, including K=2, with objective:

```text
mean negative log likelihood + (l2 / 2) * sum(W**2)
```

`l2=0.0` must be finite and nonnegative; the bias/intercept is not regularized.
Hidden layers, hidden activations and initializers are not logistic INI options.
An MLP with no hidden layers has the same softmax model family as unregularized
logistic regression, but its configured initialization can differ. Softmax output
alone does not establish calibrated clinical risk.

## Strict syntax and privacy boundary

Files use UTF-8, optionally with a BOM. Section/key names and registered values
are case-sensitive. Duplicate sections/keys, unknown or wrong-model keys, missing
`model_type`, and `[DEFAULT]` even when empty are errors. Values do not inherit
between profiles. Interpolation is disabled. Comments should be on separate
lines: inline comments are retained as part of a value, not silently removed.

Numeric tokens must be complete ASCII decimal tokens; real numbers may use
scientific notation. Integer fields reject fractional/exponent notation,
underscores and booleans. NaN, infinity, overflow and out-of-range values are
rejected. Lists reject empty interior/trailing items. Empty values have meaning
only where documented; elsewhere they are errors, not requests for defaults.
There is no file include or executable expression facility.

The driver reads clinical files, makes the patient-disjoint split and fits
imputation/scaling/encoding on training patients only. A Ray task receives numeric
arrays and numerical model settings, not source paths, patient identifiers or
raw clinical records. Numeric arrays remain sensitive: local execution is not a
de-identification or filesystem-security guarantee. Protect Ray temporary files,
object-store contents and logs, and retain sensitive split/prediction artifacts
only under the intended access and retention policy. Use Python 3.12 or 3.13 and
`uv sync --group dev`; Ray is pinned to the verified 2.60.x series. Official
[Ray tasks](https://docs.ray.io/en/latest/ray-core/tasks.html) and
[initialization](https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html)
document the remote-task/resource APIs used by the adapter.

## Python API

```python
from src.models.config import load_classifier_settings

settings = load_classifier_settings(
    "config/models.example.ini",
    model_type="multilayer_perceptron",
    profile=None,  # defaults to the canonical model type
)
assert settings.model.model_type == "multilayer_perceptron"
```

`load_classifier_settings(path, *, model_type, profile=None) -> ClassifierSettings`
returns frozen dataclasses, with `.ray: RaySettings` and `.model` one of
`MLPSettings`, `MultiOutputSettings`, `LogisticSettings`. Each model exposes a
fixed, non-constructor `model_type` field. MLP/logistic `.training` is
`TrainingSettings`; multi-output `.training` is the smaller `OnlineSettings`.
The dataclasses are immutable result records; **the loader is the validation
boundary**, not direct dataclass construction. Invalid INI syntax/settings raise
`ValueError`; file access failures retain their `OSError` subclasses. Importing
and loading this module requires only the standard library and lightweight
project contracts, not NumPy or Ray, and creates no output.

Parser tests use generated synthetic INIs and the checked-in example only:

```bash
python -m pytest tests/test_classifier_config.py -q
```

Use a dedicated scratch `TEMP`/`TMP`/`TMPDIR` per concurrent test runner so pytest
retention does not prune another runner's Ray diagnostics. Neither these parser
tests nor configuration review require opening an active clinical INI or cohort.
