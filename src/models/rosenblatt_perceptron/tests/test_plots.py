from src.models.rosenblatt_perceptron.core import train
from src.models.rosenblatt_perceptron.plots.charts import save_training_plots, save_validation_plots
from src.models.rosenblatt_perceptron.validation.metrics import evaluate

def test_epoch_history_and_csv(tmp_path):
    model, stats = train([[-2.0],[-1.0],[1.0],[2.0]],[-1,-1,1,1],epochs=20)
    history=stats["history"]
    assert len(history)==stats["epochs_run"]
    assert history[-1]["violations"]==0
    save_training_plots(history,tmp_path)
    assert (tmp_path/"epoch_history.csv").exists()

def test_holdout_plots_are_optional(tmp_path):
    y=[-1,1,-1,1]
    scores=[-2.0,2.0,-1.0,1.0]
    metrics=evaluate(y,[-1,1,-1,1],scores)
    save_validation_plots(y,scores,metrics,tmp_path)
    assert tmp_path.is_dir()
