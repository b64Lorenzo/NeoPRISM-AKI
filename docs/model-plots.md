# Rosenblatt perceptron model plots

Training automatically creates `results/models/rosenblatt_perceptron/plots/convergence/epoch_history.csv`. If matplotlib is installed, it also creates `updates_by_epoch.png`, `violations_by_epoch.png`, `accuracy_by_epoch.png`, `loss_by_epoch.png`, and `margin_by_epoch.png`.

Validation automatically creates `results/models/rosenblatt_perceptron/validation/plots/classification_metrics.png`, `roc_curve.png`, and `precision_recall_curve.png` when matplotlib is installed. ROC and precision–recall curves are only produced if both classes are present in the validation set. The confusion matrix is also saved in `validation/confusion_matrix.csv` and, when matplotlib is installed, `validation/confusion_matrix.png`.

The perceptron loss shown is the average `max(0, -y*score)` on training examples.
The `violations` count and `violations_by_epoch.png` use the configured **update**
threshold. Convergence instead requires `nonpositive_margins == 0`, meaning every
signed margin is strictly positive; the epoch CSV records both counts. A run can
converge with nonzero update-threshold violations when the threshold is positive.
The CSV's `training_accuracy` and `accuracy_by_epoch.png` report the strict
positive-margin fraction, so zero-margin ties do not count as strictly separated.
Held-out classification still assigns score-zero ties to `+1`.

Updates use fixed example order and `y*x/N`, or `y*x/(N+1)` with explicit
`--clamped` bias augmentation. See [the exact course algorithm](rosenblatt-perceptron.md).
The curves use *raw linear decision scores*, not calibrated probabilities. All
training curves are in-sample, not generalization performance; holdout validation
is separate.

Run:

```powershell
uv run python -m src.cli inputs-model --config config/inputsModel.example.ini
uv run python -m src.cli rosenblatt-train --input results/inputsModel/patients_labeled.csv --output results/models/rosenblatt_perceptron --train-fraction 0.8 --seed 2026 --max-epochs 1000
```

For PNG charts, install matplotlib (`uv add matplotlib`). Without matplotlib, model training, validation metrics, JSON, and epoch CSV still work.
