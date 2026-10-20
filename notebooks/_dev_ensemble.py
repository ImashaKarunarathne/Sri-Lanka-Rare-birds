import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"D:\Msc\Sem_2\Applied ML\Sri Lanka Rare birds")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    RandomForestClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier

RANDOM_STATE = 42

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
le_classes = sorted(y.unique())
sw_train = compute_sample_weight(class_weight="balanced", y=y_train)

print("train/test:", X_train.shape, X_test.shape)

def report(name, model, X_te, t0):
    preds = model.predict(X_te)
    acc = accuracy_score(y_test, preds)
    bal = balanced_accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro")
    print(f"{name}: acc={acc:.4f} bal_acc={bal:.4f} f1_macro={f1m:.4f}  ({time.time()-t0:.1f}s)")
    return dict(accuracy=acc, balanced_accuracy=bal, f1_macro=f1m)

results = {}

# Bagging
t0 = time.time()
param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_samples": [0.5, 0.7, 1.0],
    "max_features": [0.5, 0.7, 1.0],
    "estimator__max_depth": [5, 10, 15, None],
}
base = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
bag = BaggingClassifier(estimator=base, random_state=RANDOM_STATE, n_jobs=-1)
search = RandomizedSearchCV(bag, param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1)
search.fit(X_train, y_train)
print("Bagging best params:", search.best_params_)
results["Bagging"] = report("Bagging", search.best_estimator_, X_test, t0)
bagging_best = search.best_estimator_

# AdaBoost
t0 = time.time()
param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
    "estimator__max_depth": [1, 2, 3],
}
base = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
ada = AdaBoostClassifier(estimator=base, random_state=RANDOM_STATE)
search = RandomizedSearchCV(ada, param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1)
search.fit(X_train, y_train, sample_weight=sw_train)
print("AdaBoost best params:", search.best_params_)
results["AdaBoost"] = report("AdaBoost", search.best_estimator_, X_test, t0)
ada_best = search.best_estimator_

# Gradient Boosting
t0 = time.time()
param_dist = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.7, 0.85, 1.0],
}
gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
search = RandomizedSearchCV(gb, param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1)
search.fit(X_train, y_train, sample_weight=sw_train)
print("GradientBoosting best params:", search.best_params_)
results["GradientBoosting"] = report("GradientBoosting", search.best_estimator_, X_test, t0)
gb_best = search.best_estimator_

# XGBoost
t0 = time.time()
y_train_codes = y_train.astype("category").cat.codes
y_test_codes = pd.Categorical(y_test, categories=y_train.astype("category").cat.categories).codes
param_dist = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}
xgb = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)
search = RandomizedSearchCV(xgb, param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1)
search.fit(X_train, y_train_codes, sample_weight=sw_train)
print("XGBoost best params:", search.best_params_)
preds_codes = search.best_estimator_.predict(X_test)
acc = accuracy_score(y_test_codes, preds_codes)
bal = balanced_accuracy_score(y_test_codes, preds_codes)
f1m = f1_score(y_test_codes, preds_codes, average="macro")
print(f"XGBoost: acc={acc:.4f} bal_acc={bal:.4f} f1_macro={f1m:.4f} ({time.time()-t0:.1f}s)")
results["XGBoost"] = dict(accuracy=acc, balanced_accuracy=bal, f1_macro=f1m)
xgb_best = search.best_estimator_

# Random Forest (re-tune quickly)
t0 = time.time()
param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [10, 15, 20, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
}
rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
search = RandomizedSearchCV(rf, param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1)
search.fit(X_train, y_train)
print("RandomForest best params:", search.best_params_)
results["RandomForest"] = report("RandomForest", search.best_estimator_, X_test, t0)
rf_best = search.best_estimator_

# HistGradientBoosting (re-tune quickly)
t0 = time.time()
param_dist = {
    "max_iter": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 8, None],
    "max_leaf_nodes": [15, 31, 63],
}
hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE)
search = RandomizedSearchCV(hgb, param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1)
search.fit(X_train, y_train)
print("HistGB best params:", search.best_params_)
results["HistGradientBoosting"] = report("HistGradientBoosting", search.best_estimator_, X_test, t0)
hgb_best = search.best_estimator_

print()
print(pd.DataFrame(results).T.round(4))
