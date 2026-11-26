# Ray-backed classification baselines

The MLP, hard multi-output perceptron and multinomial logistic comparator use NumPy numerical cores executed inside **real local Ray tasks** by the shared clinical training workflow. They share patient-level splitting, preprocessing, reports and saved-model inference. The separate course Rosenblatt baseline is unchanged. The MLP follows the structure of the supplied `activation_func.py`, `nnClass.py` and `nnUtils.py`, but fixes label encoding, activation/loss validation, numerical stability and multi-output gradient scaling. It does not copy their implicit split, plotting or automatic file writes.

## Train and predict

Edit [`config/models.example.ini`](../config/models.example.ini), or copy it to
`config/models.ini`. This is the recommended place to change hidden widths,
**one activation per hidden layer**, initialization, epochs, learning rate and
local Ray resources. See [all settings and precedence](model-configuration.md).
An INI cannot be combined with architecture/training CLI overrides, even equal
ones. Input/output, target, features, split and seed stay in the CLI/XML.

```bash
uv run python -m src.cli mlp-train --config config/models.example.ini --input results/inputsModel/patients_labeled.csv --output results/models/mlp
uv run python -m src.cli multioutput-train --config config/models.example.ini --input results/inputsModel/patients_labeled.csv --output results/models/multioutput
uv run python -m src.cli logistic-train --config config/models.example.ini --input results/inputsModel/patients_labeled.csv --output results/models/logistic
```

Flag-only commands also remain available and still execute through Ray:

```bash
uv run python -m src.cli logistic-train --input results/inputsModel/patients_labeled.csv --output results/models/logistic_regression --l2 0.01
uv run python -m src.cli mlp-train --input results/inputsModel/patients_labeled.csv --output results/models/multilayer_perceptron --hidden-layers 16 8 --activation relu
uv run python -m src.cli logistic-predict --model results/models/logistic_regression/model.json --input results/inputsModel/patients_unlabeled.csv --output results/models/logistic_regression/novel_predictions.csv
uv run python -m src.cli mlp-predict --model results/models/multilayer_perceptron/model.json --input results/inputsModel/patients_unlabeled.csv --output results/models/multilayer_perceptron/novel_predictions.csv
```

MLP and logistic training commands accept `--features` (ordered column names), `--train-fraction`, `--seed`, `--max-epochs`, `--learning-rate`, `--batch-size`, and `--tolerance`. The multi-output command accepts epochs/rate, not minibatch or tolerance settings. The default target is `aki_label` with `-1/+1`. Set `--target-column aki_stage` to train against preserved clinician-reviewed stages `0,1,2,3`. A blank/unknown target is excluded, never converted to stage zero. `aki_label`, `aki_stage`, label metadata, patient IDs and `postoperative/` columns cannot be features. At least two observed classes, each with two patients, are needed for a stratified holdout. Only observed classes can be learned; probability columns follow the saved model's class order.

Numeric median imputation/scaling and categorical vocabularies are fitted on training patients only. Unknown categorical values have no active category; missing values have an explicit indicator. A fully missing training predictor is an error. Each seed yields a repeatable patient split and model fit; repeated episodes still require a separately designed group-aware split.

## Mathematics

For row samples, logits are `Z = X @ W.T + b`. Multinomial logistic regression
implements the requested conditional probability:

$$p_{θ}(Y=k\mid x)=\frac{\exp(g_k(x))}{\sum_\ell\exp(g_\ell(x))}.$$

It computes the equivalent stable expression `exp(g_k - max(g)) / sum(exp(g - max(g)))`,
normalizing over classes **per sample**, and minimizes

`mean(-log p(y_i | x_i)) + (l2 / 2) * sum(W**2)`.

Biases are not regularized. Mini-batch gradients average over the actual batch size, including the final remainder. Binary classification also uses the full two-class softmax model.

The MLP supports ReLU, leaky ReLU, sigmoid, tanh and identity hidden activations, independently per layer, with He/Xavier initialization and zero biases. Classification uses softmax/categorical cross entropy or a single sigmoid output/binary cross entropy. Losses are computed from logits. Regression uses `0.5 * mean((prediction - target)**2)` over samples **and outputs**, with consistent backpropagation. All gradients are computed before weights are updated.

## Hard multi-output perceptron

This is a genuine multiclass linear mistake learner, not a zero-hidden-layer
softmax network. Each class has `g_k(x) = w_k.T @ x + b_k`; prediction is argmax
in saved sorted-class order (ties select its first class). For a mistake, the
true row receives `+learning_rate*x` and its bias `+learning_rate`; the predicted
row receives the negatives. Other rows are unchanged. Weights/biases start at
zero, a local seeded permutation is used each epoch, and convergence requires an
entire epoch with **zero online mistakes**. Post-epoch loss is misclassification
rate, not cross entropy. Outputs are `score_<class>`, never probabilities.
Multilayer/nonlinear configurations belong to the MLP, not this hard learner.

## Python API

The following are **pure numerical kernels**, deliberately without Ray, patient
splits or file I/O. Use `classification.fit(..., model_type=..., model_config=...)`
or the CLI/XML for the Ray-orchestrated patient workflow.

```python
from src.models.logistic_regression import LogisticRegression
from src.models.neural_network import MultiLayerPerceptron

x = [[-2.0], [-1.0], [1.0], [2.0]]
y = [-1, -1, 1, 1]
model = LogisticRegression(l2=0.01, seed=2026)
model.fit(x, y, epochs=200, learning_rate=0.05, batch_size=3)
probabilities = model.predict_proba(x)  # columns match model.classes_

network = MultiLayerPerceptron([1, 8, 4, 2], ["relu", "tanh", "softmax"], seed=2026)
network.fit(x, y, epochs=200, learning_rate=0.05, batch_size=3)
restored = MultiLayerPerceptron.from_dict(network.to_dict())
assert (restored.predict(x) == network.predict(x)).all()
```

For regression, construct `MultiLayerPerceptron([input_width, hidden_width, output_width], classification=False)` and call the same `fit`/`predict` methods with numeric targets. The patient CSV CLI deliberately remains a classification workflow. Numerical classes do not split data, preprocess patients, plot or write files implicitly.

## Outputs and interpretation

- `model.json`: versioned numeric model and frozen feature transform; no patient IDs.
- `split.json`: training/validation patient IDs (**sensitive**).
- `history.csv`, `training.html`: training loss/accuracy, not validation learning curves. The hard perceptron's loss is mistake rate.
- `model.json` training metadata: actual Ray execution provenance and effective learning settings; no model INI or Ray startup is needed for inference.
- `validation/metrics.json`, `predictions.csv`: holdout evaluation.
- `validation/confusion_matrix.csv`, `.png`, `.html`: shared confusion matrix, rows true and columns predicted. Binary order `[-1,+1]`; stage order `[0,1,2,3]`, including absent classes.

Novel-data prediction exports patient IDs, predicted labels and per-class probabilities (softmax models) or unnormalized scores (hard multi-output perceptron), **not clinical truth**, and never refits preprocessing. Reports retain undefined metrics as JSON `null`. Macro metrics average defined class metrics; balanced accuracy averages class recall. The legacy binary AKI metric names are also retained.

`src/models/evaluation/` owns reusable metrics/reporting; all classifier families and Rosenblatt use it. To evaluate another classifier directly:

```python
from src.models.evaluation import save_classification_report
save_classification_report('results/evaluation', [0, 1, 2], [0, 2, 2], labels=[0, 1, 2, 3])
```

Softmax outputs are model-estimated probabilities, **not validated or calibrated clinical risks**. Perceptron decision scores remain non-probabilistic. A small single holdout is only a preliminary research result; external validation and calibration analysis remain necessary. No cohort data or actual patient performance is implied by synthetic tests.
