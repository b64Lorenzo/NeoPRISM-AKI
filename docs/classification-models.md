# Logistic regression and multilayer perceptron

These NumPy baselines share patient-level splitting, preprocessing, reports and saved-model inference. The MLP follows the structure of the supplied `activation_func.py`, `nnClass.py` and `nnUtils.py`, but fixes label encoding, activation/loss validation, numerical stability and multi-output gradient scaling. It does not copy their implicit split, plotting or automatic file writes.

## Train and predict

```bash
uv run python -m src.cli logistic-train --input results/inputsModel/patients_labeled.csv --output results/models/logistic_regression --l2 0.01
uv run python -m src.cli mlp-train --input results/inputsModel/patients_labeled.csv --output results/models/multilayer_perceptron --hidden-layers 16 8 --activation relu
uv run python -m src.cli logistic-predict --model results/models/logistic_regression/model.json --input results/inputsModel/patients_unlabeled.csv --output results/models/logistic_regression/novel_predictions.csv
uv run python -m src.cli mlp-predict --model results/models/multilayer_perceptron/model.json --input results/inputsModel/patients_unlabeled.csv --output results/models/multilayer_perceptron/novel_predictions.csv
```

Both training commands accept `--features` (ordered column names), `--train-fraction`, `--seed`, `--max-epochs`, `--learning-rate`, `--batch-size`, and `--tolerance`. The default target is `aki_label` with `-1/+1`. Set `--target-column aki_stage` to train against preserved clinician-reviewed stages `0,1,2,3`. A blank/unknown target is excluded, never converted to stage zero. `aki_label`, `aki_stage`, label metadata, patient IDs and `postoperative/` columns cannot be features. At least two observed classes, each with two patients, are needed for a stratified holdout. Only observed classes can be learned; probability columns follow the saved model's class order.

Numeric median imputation/scaling and categorical vocabularies are fitted on training patients only. Unknown categorical values have no active category; missing values have an explicit indicator. A fully missing training predictor is an error. Each seed yields a repeatable patient split and model fit; repeated episodes still require a separately designed group-aware split.

## Mathematics

For row samples, logits are `Z = X @ W.T + b`. Logistic regression uses stable softmax and minimizes

`mean(-log p(y_i | x_i)) + (l2 / 2) * sum(W**2)`.

Biases are not regularized. Mini-batch gradients average over the actual batch size, including the final remainder. Binary classification also uses the full two-class softmax model.

The MLP supports ReLU, leaky ReLU, sigmoid and identity hidden activations, with He/Xavier initialization and zero biases. Classification uses softmax/categorical cross entropy or a single sigmoid output/binary cross entropy. Losses are computed from logits. Regression uses `0.5 * mean((prediction - target)**2)` over samples **and outputs**, with consistent backpropagation. All gradients are computed before weights are updated.

## Python API

```python
from src.models.logistic_regression import LogisticRegression
from src.models.neural_network import MultiLayerPerceptron

x = [[-2.0], [-1.0], [1.0], [2.0]]
y = [-1, -1, 1, 1]
model = LogisticRegression(l2=0.01, seed=2026)
model.fit(x, y, epochs=200, learning_rate=0.05, batch_size=3)
probabilities = model.predict_proba(x)  # columns match model.classes_

network = MultiLayerPerceptron([1, 8, 2], seed=2026)
network.fit(x, y, epochs=200, learning_rate=0.05, batch_size=3)
restored = MultiLayerPerceptron.from_dict(network.to_dict())
assert (restored.predict(x) == network.predict(x)).all()
```

For regression, construct `MultiLayerPerceptron([input_width, hidden_width, output_width], classification=False)` and call the same `fit`/`predict` methods with numeric targets. The patient CSV CLI deliberately remains a classification workflow. Numerical classes do not split data, preprocess patients, plot or write files implicitly.

## Outputs and interpretation

- `model.json`: versioned numeric model and frozen feature transform; no patient IDs.
- `split.json`: training/validation patient IDs (**sensitive**).
- `history.csv`, `training.html`: training loss/accuracy, not validation learning curves.
- `validation/metrics.json`, `predictions.csv`: holdout evaluation.
- `validation/confusion_matrix.csv`, `.png`, `.html`: shared confusion matrix, rows true and columns predicted. Binary order `[-1,+1]`; stage order `[0,1,2,3]`, including absent classes.

Novel-data prediction exports patient IDs, predicted labels and per-class probabilities, **not clinical truth**, and never refits preprocessing. Reports retain undefined metrics as JSON `null`. Macro metrics average defined class metrics; balanced accuracy averages class recall. The legacy binary AKI metric names are also retained.

`src/models/evaluation/` owns reusable metrics/reporting; both new classifiers and Rosenblatt use it. To evaluate another classifier directly:

```python
from src.models.evaluation import save_classification_report
save_classification_report('results/evaluation', [0, 1, 2], [0, 2, 2], labels=[0, 1, 2, 3])
```

Softmax outputs are model-estimated probabilities, **not validated or calibrated clinical risks**. Perceptron decision scores remain non-probabilistic. A small single holdout is only a preliminary research result; external validation and calibration analysis remain necessary. No cohort data or actual patient performance is implied by synthetic tests.
