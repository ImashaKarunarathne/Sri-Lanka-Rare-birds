"""
Shared model utilities for the Sri Lanka bird habitat-guild project.

`XGBStringLabelWrapper` lives here (not inline in a notebook/script) so that
`joblib.load()` can resolve it from a *different* process -- e.g. `app.py` --
than the one that trained and pickled it. A class defined inside a notebook's
own `__main__` namespace unpickles fine within that same notebook session, but
fails everywhere else with `AttributeError: Can't get attribute ... on
<module '__main__'>`, which is exactly the failure mode a real deployment
(the app, a fresh notebook, a test script) would hit.
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBStringLabelWrapper(ClassifierMixin, BaseEstimator):
    """XGBClassifier does not accept string class labels directly; this
    label-encodes internally so `fit`/`predict`/`predict_proba` take/return
    the original string guild names, with named `__init__` params (required
    for `clone()`/`get_params()` to work inside `VotingClassifier` /
    `StackingClassifier`)."""

    def __init__(self, n_estimators=300, max_depth=4, learning_rate=0.05,
                 subsample=0.7, colsample_bytree=0.85, objective="multi:softprob",
                 eval_metric="mlogloss", random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, objective=self.objective,
            eval_metric=self.eval_metric, random_state=self.random_state, n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y_enc, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
