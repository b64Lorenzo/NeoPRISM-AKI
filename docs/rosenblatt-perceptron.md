# Binary Rosenblatt perceptron baseline

## Package layout

`src/models/rosenblatt_perceptron/` contains `core.py` (Rosenblatt update), `transform.py` (training-only fitted transforms), `training.py` (patient-disjoint stratified holdout and model export), `run.py` (CLI), `validation/metrics.py`, `validation/report.py`, `execution/predict.py` and `tests/test_model.py`.

The directory is outside preprocessing, intentionally separating clinical feature construction from model fitting, evaluation and execution. Python package name uses `rosenblatt_perceptron` rather than a folder containing spaces.

## Course algorithm

The numerical trainer follows `A1assignment/perceptron_rosenblatt.py` in
[`b64Lorenzo/nnci-course-python`](https://github.com/b64Lorenzo/nnci-course-python/blob/be038b430b599df713ee9de01ef54290306d1671/A1assignment/perceptron_rosenblatt.py),
reference commit `be038b430b599df713ee9de01ef54290306d1671`.
Patient splitting, fitted feature transforms, artifacts and reports are separate
NeoPRISM adapters, not part of the reference algorithm.

For an encoded input `x` with `N` coordinates and `y in {-1,+1}`:

1. Start all weights at zero. By default, the model is **homogeneous**: no
   additional bias input or learned intercept.
2. Visit examples in their supplied order on every epoch; never reshuffle epochs.
3. Compute the signed potential `y * sum(w[j] * x[j])`. When it is `<= threshold`,
   update each coordinate by exactly `w[j] += y * x[j] / N`.
4. Increment that example's embedding strength and the total update count once
   per update. Strengths count **training-pattern updates**, not feature importance.
5. After a complete epoch, stop if **every signed potential is strictly > 0**,
   independently of the update threshold. Otherwise continue to the epoch limit.

The threshold may be any finite number, as in the course implementation.
A positive threshold can therefore leave update-threshold violations in the
last epoch even when the run has converged. A negative threshold can prevent
all initial updates from zero weights; it is accepted for exact reference parity,
not recommended as a substitute for outcome-label definitions.

`clamped=True` explicitly copies each input to `[x[0], ..., x[N-1], 1.0]`.
The same loop then uses **N+1 coordinates and division by N+1**, including the
bias coordinate. Caller inputs are not mutated. In the patient adapter the
last learned weight is stored separately as `bias`; `weights` still match the
encoded feature-transform width. Here N counts **transformed coordinates**,
including one-hot/missingness coordinates, not raw CSV columns.

### Python API

```python
from src.models.rosenblatt_perceptron.core import train_rosenblatt

weights, epochs, updates, converged, strengths = train_rosenblatt(
    [([1.0, 0.0], 1), ([0.0, 1.0], -1)],
    max_epochs=100, threshold=0.0, clamped=False,
)
assert (weights, epochs, updates, converged, strengths) == (
    [0.5, -0.5], 1, 2, True, [1, 1],
)
```

This tuple API matches the course signature and result. `train(X, y,
*, epochs=100, threshold=0.0, clamped=False)` wraps the same training loop and
returns `(Perceptron, stats)` with epoch history. Neither numerical entry point
splits, preprocesses or shuffles data. Predictions use `+1` for score `>= 0`,
otherwise `-1`; a zero margin still fails the stricter convergence test.
Nonconvergence within a finite budget is not proof of nonseparability.


## Training and validation

```
python -m src.cli rosenblatt-train --input results/inputsModel/patients_labeled.csv --output results/models/rosenblatt_perceptron --train-fraction 0.8 --seed 2026 --max-epochs 100
```

To learn an intercept, add `--clamped`:

```bash
uv run python -m src.cli rosenblatt-train --clamped --max-epochs 100
```

`--seed` controls only the patient-disjoint split and its resulting training order;
that order stays fixed across epochs. `--threshold` controls updates, not stopping.
The default budget is 100, matching the course's **low-level trainer**; the course
study CLI's `N * nD` default concerns independent trials, not patient count, and
is not applied to this single-cohort workflow.

Unknown labels are excluded; both classes need at least two patients for a stratified holdout. Training-only numeric median imputation and standardisation, categorical one-hot encoding and missingness indicators are stored in `model.json`. Outputs include `split.json` (sensitive IDs), `validation/predictions.csv`, `validation/metrics.json`, `validation/confusion_matrix.csv`, `validation/confusion_matrix.html` (self-contained interactive Plotly), and optionally `validation/confusion_matrix.png` when matplotlib is installed. The confusion-matrix implementation is shared with other classifiers through `src/models/evaluation/`.

The confusion matrix rows are true labels, columns predicted labels, in order `[-1,+1]`:

```
[[TN, FP],
 [FN, TP]]
```

Metrics include accuracy, AKI precision/recall/F1, specificity, negative predictive value, balanced accuracy, class supports and tie-aware ROC AUC of **raw decision scores**. Undefined metrics are JSON null, not artificially set to zero. ROC AUC measures discrimination only; perceptron scores are not calibrated probabilities. With small clinical datasets, report class counts and uncertainty; a single split is preliminary.

## Execution on novel data

```
python -m src.cli rosenblatt-predict --model results/models/rosenblatt_perceptron/model.json --input results/inputsModel/patients_unlabeled.csv --output results/models/rosenblatt_perceptron/novel_predictions.csv
```

The inference command loads a frozen JSON model and its training-fitted feature transform, checks the input schema, and writes one row per patient with predicted label and linear decision score. It does **not** refit on novel patients, and does not require or output clinical outcome labels. Never use these predictions as clinical advice without external validation.

## Migration from the earlier adaptation

- New training defaults to homogeneous mode, not the previous always-learned
  bias. Use `--clamped`, `clamped=True`, or XML `<option name="clamped" value="true"/>`
  when an affine separator is required. This uses the course's N+1 normalization,
  not the previous separate-bias update with denominator N.
- Remove Rosenblatt `--learning-rate` / XML `learning_rate` settings; there is no
  rate override in the reference. Logistic-regression/MLP rate options are unchanged.
- Direct `train()` calls no longer accept `learning_rate`, `shuffle` or `seed`.
  The patient-level `fit()` retains its split seed; the shared classifier `fit()`
  rejects an explicit Rosenblatt rate rather than silently ignoring it.
- Existing schema-version-1 `model.json` files still load for prediction with their
  stored weights/bias. Loading does not retrain or reinterpret an old model as a
  course-trained one. Retrain to obtain the new semantics; previous metrics are not
  reference-parity results.

Training metadata now records `clamped`, `effective_dimensions`, fixed
`training_order`, the derived `learning_rate`, and per-pattern `strengths` in the
training order listed in `split.json`. Epoch diagnostics distinguish
`nonpositive_margins` (the convergence condition) from `violations` (the update
threshold); see [model plots](model-plots.md).

## Verification and limitations

`tests/fixtures/nnci_rosenblatt_reference.py` retains the pinned reference trainer
unchanged as a test-only oracle. Tests compare **exact results**, not rounded
weights, for multiple dimensions, epoch budgets, positive/zero/negative thresholds
and homogeneous/clamped modes. The application has no runtime dependency on the
other local repository.

The baseline remains binary. Four-stage classification is available in the
separate logistic-regression and MLP packages; see [classification models](classification-models.md).
Group-aware repeated-episode splitting, calibration and dynamic risk prediction
remain future extensions.
