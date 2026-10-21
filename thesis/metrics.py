import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score, log_loss,
    brier_score_loss, confusion_matrix,
)


def validated_predictions(y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if (y.ndim != 1 or p.ndim != 1 or not len(y) or y.shape != p.shape
            or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all()
            or ((p < 0) | (p > 1)).any()):
        raise ValueError('Expected nonempty paired binary labels and finite probabilities in [0,1].')
    return y.astype(int), p


def evaluate(y, p, threshold=0.5):
    y, p = validated_predictions(y, p)
    if not np.isfinite(threshold):
        raise ValueError('Threshold must be finite.')
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two = len(np.unique(y)) == 2
    return dict(
        n=len(y), positive_count=int(y.sum()), negative_count=int((y == 0).sum()),
        threshold=float(threshold), accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)) if two else None,
        f1=float(f1_score(y, pred, zero_division=0)),
        f1_macro=float(f1_score(y, pred, average='macro', labels=[0, 1], zero_division=0)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        specificity=float(tn / (tn + fp)) if tn + fp else None,
        loss_class_precision=float(precision_score(y, pred, pos_label=0, zero_division=0)),
        loss_class_recall=float(recall_score(y, pred, pos_label=0, zero_division=0)),
        loss_class_f1=float(f1_score(y, pred, pos_label=0, zero_division=0)),
        roc_auc=float(roc_auc_score(y, p)) if two else None,
        average_precision=float(average_precision_score(y, p)) if two else None,
        loss_class_average_precision=float(average_precision_score(1 - y, 1 - p)) if two else None,
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier=float(brier_score_loss(y, p)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def threshold_from_oof(y, p):
    """Exact BA search over attainable partitions, grouping tied probabilities.

    The decision is p >= threshold. Represent each partition by the finite
    threshold closest to 0.5. nextafter(1,+inf) represents all-negative when
    probabilities equal one. Ties prefer closeness to 0.5, then lower threshold.
    """
    y, p = validated_predictions(y, p)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        raise ValueError('OOF threshold selection requires both classes.')
    order = np.argsort(p, kind='stable')
    values, starts, counts = np.unique(p[order], return_index=True, return_counts=True)
    group_positive = np.add.reduceat(y[order], starts)
    tp = np.r_[positives, positives - np.cumsum(group_positive)]
    tn = np.r_[0, np.cumsum(counts - group_positive)]
    # Integer numerators avoid floating-point dependent ties in BA.
    numerators = tp.astype(np.int64) * negatives + tn.astype(np.int64) * positives
    candidates = np.empty(len(values) + 1)
    candidates[0] = min(0.5, values[0])
    for i in range(1, len(values)):
        lower = np.nextafter(values[i - 1], np.inf)
        candidates[i] = np.clip(0.5, lower, values[i])
    candidates[-1] = max(0.5, np.nextafter(values[-1], np.inf))
    winners = np.flatnonzero(numerators == numerators.max())
    index = min(winners, key=lambda i: (abs(candidates[i] - 0.5), candidates[i]))
    return float(candidates[index])
