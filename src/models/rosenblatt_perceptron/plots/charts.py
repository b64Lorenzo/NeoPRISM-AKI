"""Optional matplotlib diagnostics; CSV/JSON remain available without matplotlib."""
import csv
from pathlib import Path

def _mpl():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None

def save_training_plots(history, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not history: return []
    with (output/'epoch_history.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    plt=_mpl()
    if plt is None:return []
    specs=[('updates_epoch','Updates per epoch','updates_by_epoch.png'),
           ('violations','Margin violations after epoch','violations_by_epoch.png'),
           ('training_accuracy','Training accuracy','accuracy_by_epoch.png'),
           ('perceptron_loss','Mean perceptron loss','loss_by_epoch.png'),
           ('mean_signed_margin','Mean signed margin','margin_by_epoch.png')]
    paths=[]
    for key,ylabel,name in specs:
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot([r['epoch'] for r in history],[r[key] for r in history],marker='.',linewidth=1.6)
        ax.set(xlabel='Epoch',ylabel=ylabel,title=ylabel)
        ax.grid(alpha=.25);fig.tight_layout();fig.savefig(output/name,dpi=160);plt.close(fig)
        paths.append(str(output/name))
    return paths

def save_validation_plots(y_true,scores,metrics,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt=_mpl()
    if plt is None:return []
    paths=[]
    # Empirical ROC/PR use raw decision scores, not probabilities.
    if -1 in y_true and 1 in y_true:
        thresholds=[float('inf'),*sorted(set(scores),reverse=True)]
        roc=[];pr=[]
        pos=sum(v==1 for v in y_true);neg=len(y_true)-pos
        for threshold in thresholds:
            tp=sum(y==1 and s>=threshold for y,s in zip(y_true,scores))
            fp=sum(y==-1 and s>=threshold for y,s in zip(y_true,scores))
            roc.append((fp/neg,tp/pos))
            pr.append((tp/(tp+fp) if tp+fp else 1.0,tp/pos))
        for name,xy,xlabel,ylabel in [('roc_curve.png',roc,'False positive rate','True positive rate'),
                                      ('precision_recall_curve.png',pr,'Recall','Precision')]:
            fig,ax=plt.subplots(figsize=(5,5))
            ax.plot([x for x,y in xy],[y for x,y in xy],marker='.',linewidth=1.5)
            if name.startswith('roc'):ax.plot([0,1],[0,1],linestyle='--',alpha=.4)
            ax.set(xlim=(0,1),ylim=(0,1),xlabel=xlabel,ylabel=ylabel,title=name.replace('_',' ').replace('.png','').title())
            ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/name,dpi=160);plt.close(fig)
            paths.append(str(output/name))
    names=['Accuracy','Precision (AKI)','Sensitivity','Specificity','F1 (AKI)']
    keys=['accuracy','precision_aki','recall_aki_sensitivity','specificity','f1_aki']
    vals=[metrics[k] for k in keys]
    fig,ax=plt.subplots(figsize=(7,4))
    ax.barh(names,[float(v) if v is not None else 0 for v in vals]);ax.set(xlim=(0,1),xlabel='Metric value',title='Validation classification metrics')
    ax.invert_yaxis();fig.tight_layout();fig.savefig(output/'classification_metrics.png',dpi=160);plt.close(fig)
    paths.append(str(output/'classification_metrics.png'))
    return paths
