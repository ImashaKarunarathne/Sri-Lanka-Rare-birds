import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"D:\Msc\Sem_2\Applied ML\Sri Lanka Rare birds")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, VotingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier

RANDOM_STATE = 42

class XGBStringLabelWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.05,
                 subsample=0.85, colsample_bytree=0.7, objective="multi:softprob",
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

# exact tuned params found in the real notebook run
rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=1, max_features=0.5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=8, max_leaf_nodes=63, class_weight="balanced", random_state=RANDOM_STATE)
xgb = XGBStringLabelWrapper(random_state=RANDOM_STATE)
gb = GradientBoostingClassifier(subsample=0.7, n_estimators=200, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE)
bag = BaggingClassifier(estimator=DecisionTreeClassifier(class_weight="balanced", max_depth=15, random_state=RANDOM_STATE),
                         n_estimators=200, max_samples=1.0, max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1)

sw_train = compute_sample_weight(class_weight="balanced", y=y_train)

def report(name, model):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    bal = balanced_accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro")
    print(f"{name}: acc={acc:.4f} bal_acc={bal:.4f} f1_macro={f1m:.4f}")
    return f1m

t0=time.time()
rf.fit(X_train, y_train)
hgb.fit(X_train, y_train)
xgb.fit(X_train, y_train)
gb.fit(X_train, y_train, sample_weight=sw_train)
bag.fit(X_train, y_train)
print("base fits:", time.time()-t0)

report("RF alone", rf)
report("Bagging alone", bag)

# Try several ensemble variants
variants = {}

variants["Voting(RF,HGB,XGB)"] = VotingClassifier(estimators=[("rf",rf),("hgb",hgb),("xgb",xgb)], voting="soft", n_jobs=-1)
variants["Voting(RF,Bag,HGB,XGB)"] = VotingClassifier(estimators=[("rf",rf),("bag",bag),("hgb",hgb),("xgb",xgb)], voting="soft", n_jobs=-1)
variants["Voting(RF,Bag)"] = VotingClassifier(estimators=[("rf",rf),("bag",bag)], voting="soft", n_jobs=-1)
variants["Stacking(RF,HGB,XGB)"] = StackingClassifier(estimators=[("rf",rf),("hgb",hgb),("xgb",xgb)], final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE), cv=5, n_jobs=-1)
variants["Stacking(RF,Bag,HGB,XGB)"] = StackingClassifier(estimators=[("rf",rf),("bag",bag),("hgb",hgb),("xgb",xgb)], final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE), cv=5, n_jobs=-1)
variants["Stacking(RF,Bag,HGB,XGB,GB)+passthrough"] = StackingClassifier(estimators=[("rf",rf),("bag",bag),("hgb",hgb),("xgb",xgb),("gb",gb)], final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE), cv=5, n_jobs=-1, passthrough=True)
variants["Voting(RF,Bag,HGB,XGB,GB) weighted"] = VotingClassifier(estimators=[("rf",rf),("bag",bag),("hgb",hgb),("xgb",xgb),("gb",gb)], voting="soft", weights=[2,2,1,1,1], n_jobs=-1)

results = {}
for name, model in variants.items():
    t0=time.time()
    model.fit(X_train, y_train)
    f1m = report(name, model)
    results[name] = f1m
    print(f"  ({time.time()-t0:.1f}s)")

print()
print("Best individual: RF=0.7269 or Bagging=0.7276 (from the real notebook run)")
for k,v in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"{v:.4f}  {k}")
