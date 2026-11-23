import csv
import json
from pathlib import Path
from src.models.rosenblatt_perceptron.validation.metrics import evaluate
from src.models.rosenblatt_perceptron.plots.charts import save_validation_plots

def save_report(path:Path,patient_ids,y_true,y_pred,scores):
    path.mkdir(parents=True,exist_ok=True)
    report=evaluate(y_true,y_pred,scores)
    (path/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    with (path/'predictions.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['patient_id','true_label','predicted_label','decision_score'])
        w.writerows(zip(patient_ids,y_true,y_pred,scores))
    with (path/'confusion_matrix.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['true/predicted','No AKI (-1)','AKI (+1)'])
        w.writerow(['No AKI (-1)',*report['confusion_matrix'][0]])
        w.writerow(['AKI (+1)',*report['confusion_matrix'][1]])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(5,4))
        ax.imshow(report['confusion_matrix'],cmap='Blues')
        for i,row in enumerate(report['confusion_matrix']):
            for j,n in enumerate(row):ax.text(j,i,str(n),ha='center',va='center')
        ax.set_xticks([0,1],['No AKI','AKI']);ax.set_yticks([0,1],['No AKI','AKI'])
        ax.set_xlabel('Predicted');ax.set_ylabel('True');fig.tight_layout()
        fig.savefig(path/'confusion_matrix.png',dpi=180);plt.close(fig)
    except ImportError: pass
    save_validation_plots(y_true,scores,report,path/'plots')
    return report
