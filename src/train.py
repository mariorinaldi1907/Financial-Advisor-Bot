"""Shared model primitives. Run this module to train the current research system."""
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]


class PriorModel:
    """Smoothed per-asset prior, kept in an importable module for portable artifacts."""
    def fit(self, X, y):
        self.columns = [c for c in X if c.startswith('Asset_')]
        self.rates = np.array([(np.asarray(y)[X[c].to_numpy() == 1].sum() + 1) /
                               (int((X[c] == 1).sum()) + 2) for c in self.columns])
        return self

    def predict_proba(self, X):
        p = np.clip(X[self.columns].to_numpy() @ self.rates, 1e-6, 1 - 1e-6)
        return np.column_stack([1-p, p])


def scores(y, p):
    pred = (np.asarray(p) >= .5).astype(int)
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                brier=float(brier_score_loss(y, p)),
                roc_auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def purged_before(frame, boundary):
    return frame.loc[frame.Label_end < boundary]


if __name__ == '__main__':
    from .research import run
    run()
