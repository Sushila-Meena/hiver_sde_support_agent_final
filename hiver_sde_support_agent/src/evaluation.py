import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


def intent_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def pearson_or_none(x, y):
    if len(x) < 2:
        return None
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])
