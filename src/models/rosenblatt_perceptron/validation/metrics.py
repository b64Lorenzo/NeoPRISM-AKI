"""Binary classification metrics, with -1 = No AKI and +1 = AKI."""
import math

def evaluate(y_true,y_pred,scores=None):
    if not y_true or len(y_true)!=len(y_pred): raise ValueError('Mismatched or empty predictions')
    if any(v not in (-1,1) for v in [*y_true,*y_pred]): raise ValueError('Labels must be -1 or +1')
    tn=sum(a==-1 and b==-1 for a,b in zip(y_true,y_pred))
    fp=sum(a==-1 and b==1 for a,b in zip(y_true,y_pred))
    fn=sum(a==1 and b==-1 for a,b in zip(y_true,y_pred))
    tp=sum(a==1 and b==1 for a,b in zip(y_true,y_pred))
    ratio=lambda a,b: a/b if b else None
    precision=ratio(tp,tp+fp);recall=ratio(tp,tp+fn)
    f1=ratio(2*tp,2*tp+fp+fn)
    specificity=ratio(tn,tn+fp)
    balanced=(recall+specificity)/2 if recall is not None and specificity is not None else None
    out={'labels':[-1,1],'confusion_matrix':[[tn,fp],[fn,tp]],'tn':tn,'fp':fp,'fn':fn,'tp':tp,
         'accuracy':(tp+tn)/len(y_true),'precision_aki':precision,'recall_aki_sensitivity':recall,
         'specificity':specificity,'f1_aki':f1,'balanced_accuracy':balanced,
         'negative_predictive_value':ratio(tn,tn+fn),'support_no_aki':tn+fp,'support_aki':tp+fn}
    if scores is not None:
        out['roc_auc']=auc(y_true,scores)
    return out

def auc(y,scores):
    """Tie-aware rank AUC of raw linear scores, undefined if only one class."""
    if len(y)!=len(scores):raise ValueError('Score length mismatch')
    positives=sum(v==1 for v in y);negatives=sum(v==-1 for v in y)
    if not positives or not negatives:return None
    if any(not math.isfinite(float(v)) for v in scores):raise ValueError('Nonfinite scores')
    ordered=sorted(zip(scores,y));sum_positive_ranks=0.0;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        average_rank=(i+1+j)/2
        sum_positive_ranks+=sum(v==1 for _,v in ordered[i:j])*average_rank
        i=j
    return (sum_positive_ranks-positives*(positives+1)/2)/(positives*negatives)
