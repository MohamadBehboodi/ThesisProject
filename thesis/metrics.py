import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix
)

def evaluate(y,p,threshold=0.5):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)

    if len(y)!=len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probabilities.')
    
    pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    two=len(np.unique(y))==2

    return dict(
        n=len(y),
        positive_count=int(y.sum()),
        threshold=float(threshold),
        accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)) if two else None,
        f1=float(f1_score(y,pred,zero_division=0)),
        f1_macro=float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
        precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)),
        specificity=float(tn/(tn+fp)) if tn+fp else None,
        roc_auc=float(roc_auc_score(y,p)) if two else None,
        average_precision=float(average_precision_score(y,p)) if two else None,
        log_loss=float(log_loss(y,p,labels=[0,1])),
        brier=float(brier_score_loss(y,p)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp)
    )

def threshold_from_oof(y,p):
    ts=np.linspace(0.05,0.95,181)
    best=max(ts,key=lambda t:(balanced_accuracy_score(y,np.asarray(p)>=t),-abs(t-.5)))
    return float(best)
