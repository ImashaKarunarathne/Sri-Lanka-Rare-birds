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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import shap
import lime
import lime.lime_tabular

RANDOM_STATE = 42

class XGBStringLabelWrapper(ClassifierMixin, BaseEstimator):
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

rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=1, max_features=0.5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=8, max_leaf_nodes=63, class_weight="balanced", random_state=RANDOM_STATE)
xgb = XGBStringLabelWrapper(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.7, colsample_bytree=0.85, random_state=RANDOM_STATE, n_jobs=-1)

t0=time.time()
rf.fit(X_train, y_train); hgb.fit(X_train, y_train); xgb.fit(X_train, y_train)
print("base fits done", time.time()-t0)

t0=time.time()
stack = StackingClassifier(
    estimators=[("rf", rf), ("hgb", hgb), ("xgb", xgb)],
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    cv=5, n_jobs=-1,
)
stack.fit(X_train, y_train)
print("stack fit done", time.time()-t0)

# ---- SHAP on XGBoost (tree explainer, exact + fast for multiclass) ----
t0 = time.time()
explainer = shap.TreeExplainer(xgb.model_)
X_sample = X_test.sample(n=300, random_state=RANDOM_STATE)
shap_values = explainer.shap_values(X_sample)
print("shap_values type:", type(shap_values), getattr(shap_values, "shape", None))
print("shap took", time.time()-t0)

mean_abs = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs)[::-1][:10]
print("Top 10 features by mean |SHAP| across classes:")
for i in top_idx:
    print(f"  {X_sample.columns[i]:30s} {mean_abs[i]:.4f}")

# ---- LIME on final stacking ensemble (model-agnostic) ----
t0 = time.time()
class_names = list(xgb.classes_)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(X_train.columns),
    class_names=class_names,
    mode="classification",
    random_state=RANDOM_STATE,
)
idx = 0
inst = X_test.iloc[idx].values
true_label = y_test.iloc[idx]
exp = lime_explainer.explain_instance(inst, stack.predict_proba, num_features=8, top_labels=1)
print("LIME done", time.time()-t0, "true_label=", true_label)
print(exp.available_labels())
print(exp.as_list(label=exp.available_labels()[0]))
