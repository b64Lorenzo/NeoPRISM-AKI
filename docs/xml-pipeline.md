# XML pipeline

XML is an optional orchestration layer, not a replacement for the parameter,
plotting, or model-input INIs. Existing commands keep their current interfaces.
The distributed `config/pipeline.xml` and `config/pipeline.example.xml` are safe,
disabled starters: loading or running either reads no domain files and creates no
output directories.

## Using a configuration

From the repository root, run:

```bash
uv run python -m src.cli pipeline --config config/pipeline.xml
```

The Python entry point works independently of that CLI registration:

```python
from src.pipeline.run import run_pipeline

status = run_pipeline("config/pipeline.xml")
```

Edit a copy of the starter. Replace placeholder input paths and `feature_1` /
`feature_2` with the **exact CSV headers** in each selected matrix. Review the
referenced INIs before enabling their stages; example INIs may themselves refer
to other configurations that need adjustment. Enable only the desired `run`
flags. XML booleans accept `true`, `false`, `1`, and `0`; omission means false.

```mermaid
flowchart LR
    XML[Validate complete XML] --> E[Optional extraction]
    E --> P[Optional plotting]
    P --> I[Optional inputsModel]
    I --> M[Enabled models in XML order]
```

No stage automatically inherits another stage's paths. Set the plotting/input
INIs to consume the extraction output when that is intended. Set each model's
`inputMatrix` explicitly to the desired matrix, whether preprocessing creates it
in this invocation or it already exists. `unlabeled="true"` maps to the existing
`inputs-model --unlabeled` behavior; training still needs a matrix containing its
requested target.

## XML contract

There is one unnamespaced root, `pipelineConfiguration`, with optional singleton
children `preprocessing` followed by `models`; there is no `general` wrapper.
Empty roots, empty sections, preprocessing-only, and model-only documents work.

Within `preprocessing`, optional singleton stages must be declared in this order:

| Element | Required settings | Optional settings |
| --- | --- | --- |
| `extraction` | `config`, `outputDir` attributes; one or more `input` text children | `run` |
| `plotting` | `config` attribute | `run` |
| `inputsModel` | `config` attribute | `run`, `unlabeled` |

The runner wraps existing `src.preprocessing.parameters.run.run`,
`src.preprocessing.plotting.run.run`, and
`src.preprocessing.inputsModel.run.run` with explicit argument namespaces.
Imports occur only inside enabled handlers. `csv-header` remains standalone.

`models` contains zero or more ordered `model` entries. Repeating a type is
supported; identifiers must be unique. Each entry requires `id` and `type`
attributes, optional `run`, and these children **in order**:

1. `inputMatrix`: this model's CSV path.
2. `outputDir`: this model's output directory.
3. `targetColumn`: this model's target header.
4. `features`: one or more ordered `column` text children.
5. `split`: required `trainFraction` and `seed` attributes.
6. `options`: zero or more empty `option` elements with required `name` and
   `value` attributes. An empty `options` container selects model-specific defaults.

IDs match `[A-Za-z_][A-Za-z0-9_.-]*`; type/option names match
`[a-z][a-z0-9_]*`. Column names preserve case and internal whitespace; surrounding
whitespace is trimmed. Feature columns must be unique and exclude the selected
target. Built-in models also reject patient IDs, either AKI outcome column,
label metadata and `postoperative/` predictors during XML loading. The training
fraction must be finite and strictly between zero and one;
seeds are nonnegative decimal integers. Unknown attributes/elements, mixed text,
missing/duplicate singleton nodes, and out-of-order nodes are errors.

All **XML-written paths** resolve against the directory containing that XML
file, not the working directory. Absolute paths remain absolute. No environment
variable or tilde expansion is performed. Paths *inside INIs* keep the existing
INI loader semantics; XML does not rebase them, rewrite files, or change cwd.
Moving an XML file therefore requires reviewing its relative paths.

## Built-in model options

Names below are Python keyword names, not CLI flag spellings. Numeric values
must be finite; epochs/batch sizes/layer widths are positive integers. Rates are
strictly positive; tolerance and L2 are nonnegative.

| Type | Targets | Options and defaults |
| --- | --- | --- |
| `rosenblatt_perceptron` | `aki_label` only | `epochs=100`, omitted `learning_rate=None`, `threshold=0` |
| `logistic_regression` | `aki_label`, `aki_stage` | `epochs=1000`, `learning_rate=0.05`, `batch_size=32`, `tolerance=0`, `l2=0` |
| `multilayer_perceptron` | `aki_label`, `aki_stage` | `epochs=1000`, `learning_rate=0.05`, `batch_size=32`, `tolerance=0`, `hidden_layers=(16,)`, `activation=relu` |

`hidden_layers` uses comma-separated positive widths, for example `32,16`.
`activation` accepts `relu`, `leaky_relu`, `sigmoid`, `tanh`, or `linear`, matching
the standalone MLP CLI. Rosenblatt thresholds must be nonnegative. Its automatic learning
rate is selected by **omitting** that option, not by writing the string `None`.
Options not listed for that particular type are rejected. MLP has no `l2` option;
Rosenblatt has no minibatch or tolerance option.

Built-in execution imports `src.models.classification.fit` only when a model is
enabled and its matrix exists, then calls exactly:

```python
fit(
    config.input_matrix,
    config.output_dir,
    model_type=config.model_type,
    target_column=config.target_column,
    features=config.features,
    train_fraction=config.train_fraction,
    seed=config.seed,
    **options,
)
```

The returned training-result dictionary belongs to the classification backend;
the pipeline returns status zero after a successful call and does not print or
persist that dictionary itself. Real training therefore requires that backend;
disabled configurations and custom registries do not import it.

## Adding a model without changing the loader

`ModelRegistry.register(type_name, parse_options, handler)` binds a typed option
parser to its matching handler. The parser signature is
`(ModelConfig, Mapping[str, str]) -> T`; the handler is
`(ModelConfig, T) -> int | None`. `ModelConfig` is immutable and exposes `id`,
`model_type`, `input_matrix`, `output_dir`, `target_column`, ordered `features`,
`train_fraction`, `seed`, and `run`.

```python
from dataclasses import dataclass
from collections.abc import Mapping
from src.pipeline.config import ModelConfig, parse_integer
from src.pipeline.registry import create_default_registry
from src.pipeline.run import run_pipeline

@dataclass(frozen=True)
class TreeOptions:
    depth: int

def parse_tree(config: ModelConfig, raw: Mapping[str, str]) -> TreeOptions:
    if raw.keys() - {"depth"}:
        raise ValueError("Unknown tree option")
    if config.target_column not in {"aki_label", "aki_stage"}:
        raise ValueError("Unsupported tree target")
    return TreeOptions(parse_integer(raw.get("depth", "3"), "depth", minimum=1))

def run_tree(config: ModelConfig, options: TreeOptions) -> int:
    # Import your backend here and train using config's explicit paths/schema.
    from my_project.tree_backend import fit_tree
    fit_tree(config.input_matrix, config.output_dir,
             target=config.target_column, features=config.features,
             train_fraction=config.train_fraction, seed=config.seed,
             depth=options.depth)
    return 0

registry = create_default_registry()
registry.register("decision_tree", parse_tree, run_tree)
status = run_pipeline("my-pipeline.xml", registry=registry)
```

The new XML entry uses `type="decision_tree"` and, for example,
`<option name="depth" value="5"/>`. No loader, dispatch switch, or XSD enum needs
to change. Use `ModelRegistry()` instead for a registry containing only your own
types. Duplicate registrations and missing/noncallable handlers are rejected.
The same supplied registry is used for parsing and execution: each `ModelStep`
captures its validated options and matching handler. Option parsers must be
pure (including for disabled entries); reject unsupported settings and return
immutable typed options. Backend imports, file reads, and output creation belong
in the handler, not the parser.

## Validation, failures, and schema

`load_config(path, registry=None)` returns immutable `PipelineConfig` with
`preprocessing` and an ordered tuple of prepared `models`; each model exposes
its common configuration as `.config`. Built-in options are read-only mappings.
All XML entries and option parsers are validated **before any handler executes**,
including disabled entries. Unknown model types are errors, not silently skipped.
Referenced INI content, matrix headers, and training data are validated by their
own enabled domain handlers, not by opening them during XML parsing.

Enabled model output directories must be distinct after path resolution; this
check does not inspect plotting/input-building outputs hidden inside INIs.
Matrix existence is checked immediately before each enabled model, after
preprocessing. Failures/exceptions stop later work; nonzero handler statuses are
returned unchanged. Execution is sequential, not transactional: completed earlier
stages are not rolled back, and the framework is not an output sandbox. Review
enabled paths before running; backend overwrite policies still apply.

The loader reads at most `MAX_XML_BYTES + 1` bytes (default ceiling: 1 MiB), rejects
oversize input and DTDs using an ElementTree `doctype` callback, and does not
process external schemas or XInclude. The callback catches UTF-16 declarations as
well as UTF-8. XML comments are allowed; processing instructions are rejected.
Malformed XML is reported as `ValueError` for compatibility with the root CLI.
`xsi:noNamespaceSchemaLocation` is an optional editor hint only, never fetched.

`config/pipeline.xsd` is an optional editor/validation schema. Runtime validation
uses only the Python standard library and does **not** require lxml. XSD enforces
structure/order, basic values, and identifier/feature/option uniqueness; plugin
registration, model-target/options compatibility, numeric finiteness, and resolved
enabled-output checks remain runtime responsibilities. Types stay extensible.

Optional XSD validation without adding a project dependency:

```bash
uv run --no-project --python 3.12 --with lxml python -c 'from lxml import etree; s=etree.XMLSchema(etree.parse("config/pipeline.xsd")); [s.assertValid(etree.parse(p)) for p in ("config/pipeline.xml", "config/pipeline.example.xml")]; print("Both XML templates validate")'
```

Focused tests use only generated synthetic fixtures:

```bash
uv run python -m pytest tests/test_pipeline.py -q
```

The XSD-specific test skips if lxml is absent. Training tests should use synthetic
matrices and the shared backend; the framework tests also exercise real custom
registry handlers that consume synthetic CSVs and create temporary artifacts.
Do not enable the active configuration as a test fixture or run against private
data/results. Set `TEMP`, `TMP`, and `TMPDIR` to an isolated scratch directory
before running verification commands.

References: Python's [ElementTree API](https://docs.python.org/3.12/library/xml.etree.elementtree.html#treebuilder-objects)
and [XML security notes](https://docs.python.org/3.12/library/xml.html#xml-vulnerabilities).
