import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"D:\Msc\Sem_2\Applied ML\Sri Lanka Rare birds")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

RANDOM_STATE = 42

class XGBStringLabelWrapper(ClassifierMixin, BaseEstimator):
    """Wraps XGBClassifier so it can take/return string labels, like every other
    sklearn classifier here -- needed so it can sit inside VotingClassifier /
    StackingClassifier next to RandomForest/HistGradientBoosting without every
    other piece of code having to special-case it. Named __init__ params
    (rather than **kwargs) are required so sklearn's clone()/get_params() work
    correctly -- StackingClassifier clones base estimators internally for its
    cross-validated meta-features. Also self-balances via sample_weight since
    XGBClassifier has no class_weight param."""
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
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        y_enc = self.le_.transform(y)
        if sample_weight is None:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
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

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bird_habitat_guild_classification.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["eventDate"])

numeric_cols = [
    "decimalLatitude", "decimalLongitude", "avg_rad", "NDVI", "elevation_meters",
    "Carbon_Mass", "Dust_Mass", "SO2_Mass", "Sulfate_Mass", "Sea_Salt_Mass",
    "Total_Aerosol_Extinction", "temp_mean", "rainfall", "wind_mean",
    "humid_mean", "shortwave_radiation", "lka_general_2020",
]
categorical_cols = ["stateProvince", "LandCover_Class"]
X = pd.get_dummies(df[numeric_cols + categorical_cols], columns=categorical_cols, drop_first=True)
y = df["habitat_guild"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
sw_train = compute_sample_weight(class_weight="balanced", y=y_train)

rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=1, max_features=0.5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=8, max_leaf_nodes=63, class_weight="balanced", random_state=RANDOM_STATE)
xgb = XGBStringLabelWrapper(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.7, colsample_bytree=0.85, random_state=RANDOM_STATE, n_jobs=-1)

def report(name, model, t0):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    bal = balanced_accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro")
    print(f"{name}: acc={acc:.4f} bal_acc={bal:.4f} f1_macro={f1m:.4f} ({time.time()-t0:.1f}s)")
    return dict(accuracy=acc, balanced_accuracy=bal, f1_macro=f1m)

results = {}

t0=time.time(); rf.fit(X_train, y_train); results["RandomForest (tuned)"] = report("RandomForest (tuned)", rf, t0)
t0=time.time(); hgb.fit(X_train, y_train); results["HistGB (tuned)"] = report("HistGB (tuned)", hgb, t0)
t0=time.time(); xgb.fit(X_train, y_train); results["XGBoost (tuned)"] = report("XGBoost (tuned)", xgb, t0)

# Voting ensemble (soft)
t0 = time.time()
voting = VotingClassifier(estimators=[("rf", rf), ("hgb", hgb), ("xgb", xgb)], voting="soft", n_jobs=-1)
voting.fit(X_train, y_train)
results["Voting Ensemble"] = report("Voting Ensemble", voting, t0)

# Stacking ensemble
t0 = time.time()
stack = StackingClassifier(
    estimators=[("rf", rf), ("hgb", hgb), ("xgb", xgb)],
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    cv=5, n_jobs=-1, passthrough=False,
)
stack.fit(X_train, y_train)
results["Stacking Ensemble"] = report("Stacking Ensemble", stack, t0)

print()
print(pd.DataFrame(results).T.round(4))

print()
print(classification_report(y_test, stack.predict(X_test)))
