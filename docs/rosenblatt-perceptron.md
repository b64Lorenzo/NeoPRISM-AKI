# Binary Rosenblatt perceptron baseline

## Package layout

`src/models/rosenblatt_perceptron/` contains `core.py` (Rosenblatt update), `transform.py` (training-only fitted transforms), `training.py` (patient-disjoint stratified holdout and model export), `run.py` (CLI), `validation/metrics.py`, `validation/report.py`, `execution/predict.py` and `tests/test_model.py`.

The directory is outside preprocessing, intentionally separating clinical feature construction from model fitting, evaluation and execution. Python package name uses `rosenblatt_perceptron` rather than a folder containing spaces.

## Mathematics

For transformed vector `x`, the linear score is `s(x) = w·x + b`, with prediction `+1` when `s(x) >= 0`, otherwise `-1`. For a training example `(x,y)` with `y in {-1,+1}`, if `y*s(x) <= c`, update `w <- w + eta*y*x`, `b <- b + eta*y`. The default learning rate is `1/(encoded dimension)` and `c=0`. Epochs stop when all examples have strictly positive signed margin above the threshold or the finite epoch budget is exhausted. With nonseparable data, convergence is not guaranteed. The intercept corresponds to the clamped-input extension of the NNCI reference.

## Training and validation

```
python -m src.cli rosenblatt-train --input results/inputsModel/patients_labeled.csv --output results/models/rosenblatt_perceptron --train-fraction 0.8 --seed 2026 --max-epochs 100
```

Unknown labels are excluded; both classes need at least two patients for a stratified holdout. Training-only numeric median imputation and standardisation, categorical one-hot encoding and missingness indicators are stored in `model.json`. Outputs include `split.json` (sensitive IDs), `validation/predictions.csv`, `validation/metrics.json`, `validation/confusion_matrix.csv`, and optionally `validation/confusion_matrix.png` when matplotlib is installed.

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

## References and limitations

Implementation is adapted conceptually from the user's Rosenblatt assignment (https://github.com/b64Lorenzo/nnci-course-python) and the modular `prism-ml` package (https://github.com/b64Lorenzo/neoprism), rather than a verbatim copy. The baseline is binary; four-stage KDIGO classification, probabilistic logistic regression, group-aware repeated-episode splitting, calibration and dynamic risk prediction are future extensions.
