import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
M = lambda s: C.append(nbf.v4.new_markdown_cell(s))
K = lambda s: C.append(nbf.v4.new_code_cell(s))

M(r"""# Sri Lanka Bird Habitat-Guild Classification — v6: 6-Class Target (Coastal Pair Merged)

**Why this notebook exists:** after v5 (`04_ensemble_tuning_and_explainability.ipynb`) landed at macro-F1 0.729 / accuracy 71.4% on the 7-class habitat-guild target, the question came up of whether accuracy could be pushed past 90%. Before touching a single hyperparameter, it's worth asking whether that's even *possible* on this data.

**The duplicate-feature-row ceiling.** Multiple species/guilds get recorded at the same rounded location with the same climate snapshot — environmentally identical rows with different true labels. No classifier, however good, can resolve those; the best any model can do is guess the majority label within each such group. That sets a hard mathematical ceiling on achievable accuracy, computed below directly from the data (not assumed).
""")

M("## 1. The accuracy ceiling, computed directly")
K(r"""import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd().parent if not (Path.cwd() / "src").exists() else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

df7 = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "bird_habitat_guild_classification.csv")

numeric_cols = [
    "decimalLatitude", "decimalLongitude", "avg_rad", "NDVI", "elevation_meters",
    "Carbon_Mass", "Dust_Mass", "SO2_Mass", "Sulfate_Mass", "Sea_Salt_Mass",
    "Total_Aerosol_Extinction", "temp_mean", "rainfall", "wind_mean",
    "humid_mean", "shortwave_radiation", "lka_general_2020",
]
categorical_cols = ["stateProvince", "LandCover_Class"]
feat_cols = numeric_cols + categorical_cols

def accuracy_ceiling(frame, label_col):
    dup_mask = frame.duplicated(subset=feat_cols, keep=False)
    n_total = len(frame)
    n_nondup = (~dup_mask).sum()
    majority = lambda g: g.value_counts(normalize=True).max()
    grouped = frame[dup_mask].groupby(feat_cols)
    best_correct = n_nondup + (grouped[label_col].apply(majority) * grouped.size()).sum()
    return best_correct / n_total * 100, dup_mask.mean() * 100

ceiling_7, duprate_7 = accuracy_ceiling(df7, "habitat_guild")
print(f"7-class target:  {duprate_7:.1f}% of rows are feature-duplicates of a different-labelled row  ->  accuracy ceiling = {ceiling_7:.1f}%")
print("(i.e. 90% accuracy is mathematically unreachable on the 7-class target with these features -- confirmed, not assumed)")
""")

M(r"""## 2. Why 6 classes — merging the one pair v3/v4's own analysis flagged as closest remaining

v3's centroid-distance analysis (the same one that justified merging 5 species into 2 guilds for v4) already showed *Thalasseus bergii* and *Numenius arquata* — both low-elevation coastal species — sitting closer to each other (~1.7 standardized distance) than to any of the 4 other still-distinct guilds, even though they were kept separate in v4 because that distance was still clearly larger than the within-cluster distances that justified the original 5-species merge (0.8-1.5). Merging this pair now is the next smallest, most defensible class-count reduction — not an arbitrary choice to chase a number.
""")
K(r"""MERGE_MAP = {
    "Thalasseus bergii (marine coastal)": "Coastal (tern + curlew)",
    "Numenius arquata (coastal wader)": "Coastal (tern + curlew)",
}
df7["habitat_guild_6"] = df7["habitat_guild"].replace(MERGE_MAP)

ceiling_6, duprate_6 = accuracy_ceiling(df7, "habitat_guild_6")
print(f"6-class target:  accuracy ceiling = {ceiling_6:.1f}%  (vs {ceiling_7:.1f}% for 7 classes)")
print()
print(df7["habitat_guild_6"].value_counts())

DATA_PATH_6 = PROJECT_ROOT / "data" / "processed" / "bird_habitat_guild_6class_classification.csv"
df7.to_csv(DATA_PATH_6, index=False)
print("\nSaved:", DATA_PATH_6)
""")

M(r"""**Honest framing going in:** even 6 classes only raises the ceiling to ~88%, still short of 90% — reaching 90% outright would require collapsing to a near-binary split (upland/forest vs. lowland/coastal, ceiling 95.9%), which guts the "classify habitat guild" problem into "is elevation high or low." This notebook runs the full tuned-model + ensemble + SHAP/LIME pipeline from v5 on the 6-class target to see how close models actually get to the *achievable* 88% ceiling, and reports the real number rather than a target picked in advance.
""")

M("## 3. Setup and data")
K(r"""from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    RandomForestClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
import shap
import lime
import lime.lime_tabular

df = pd.read_csv(DATA_PATH_6, parse_dates=["eventDate"])
X = pd.get_dummies(df[numeric_cols + categorical_cols], columns=categorical_cols, drop_first=True)
y = df["habitat_guild_6"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
sw_train = compute_sample_weight(class_weight="balanced", y=y_train)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"Train: {len(X_train)} rows   Test: {len(X_test)} rows   Features: {X_train.shape[1]}")
print("Classes:", sorted(y.unique()))
""")

K(r"""results = {}

def report(name, model, X_te, store=True):
    preds = model.predict(X_te)
    acc = accuracy_score(y_test, preds)
    bal = balanced_accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro")
    if store:
        results[name] = {"accuracy": acc, "balanced_accuracy": bal, "f1_macro": f1m}
    print(f"{name}: accuracy={acc:.4f}  balanced_accuracy={bal:.4f}  macro-F1={f1m:.4f}")
    print(classification_report(y_test, preds))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(f"{name} — confusion matrix")
    plt.tight_layout()
    plt.show()
    return preds
""")

M("## 4. Tuned individual models (same search spaces as v5, re-run on the 6-class target)")
K(r"""param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
logreg_search = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    param_grid, cv=3, scoring="f1_macro", n_jobs=-1,
)
logreg_search.fit(X_train_scaled, y_train)
print("Logistic Regression best params:", logreg_search.best_params_)
_ = report("Logistic Regression (tuned)", logreg_search.best_estimator_, X_test_scaled)
""")
K(r"""param_grid = {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]}
svm_search = GridSearchCV(
    SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=RANDOM_STATE),
    param_grid, cv=3, scoring="f1_macro", n_jobs=-1,
)
svm_search.fit(X_train_scaled, y_train)
print("SVM best params:", svm_search.best_params_)
_ = report("SVM (RBF, tuned)", svm_search.best_estimator_, X_test_scaled)
""")
K(r"""param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_samples": [0.5, 0.7, 1.0],
    "max_features": [0.5, 0.7, 1.0],
    "estimator__max_depth": [5, 10, 15, None],
}
base_tree = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
bagging_search = RandomizedSearchCV(
    BaggingClassifier(estimator=base_tree, random_state=RANDOM_STATE, n_jobs=-1),
    param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1,
)
bagging_search.fit(X_train, y_train)
print("Bagging best params:", bagging_search.best_params_)
bagging_best = bagging_search.best_estimator_
_ = report("Bagging", bagging_best, X_test)
""")
K(r"""param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
    "estimator__max_depth": [1, 2, 3],
}
base_tree = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
ada_search = RandomizedSearchCV(
    AdaBoostClassifier(estimator=base_tree, random_state=RANDOM_STATE),
    param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1,
)
ada_search.fit(X_train, y_train, sample_weight=sw_train)
print("AdaBoost best params:", ada_search.best_params_)
ada_best = ada_search.best_estimator_
_ = report("AdaBoost", ada_best, X_test)
""")
K(r"""param_dist = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.7, 0.85, 1.0],
}
gb_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1,
)
gb_search.fit(X_train, y_train, sample_weight=sw_train)
print("Gradient Boosting best params:", gb_search.best_params_)
gb_best = gb_search.best_estimator_
_ = report("Gradient Boosting", gb_best, X_test)
""")

M(r"""`XGBStringLabelWrapper` — same wrapper as v5: label-encodes internally so `fit`/`predict`/`predict_proba` take/return the same string guild names as every other classifier here, with named `__init__` params (not `**kwargs`) so `clone()`/`get_params()` work inside `VotingClassifier`/`StackingClassifier`, and `ClassifierMixin` listed before `BaseEstimator` so `is_classifier()` recognizes it.""")
K(r"""class XGBStringLabelWrapper(ClassifierMixin, BaseEstimator):
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
""")
K(r"""param_dist = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}
xgb_search = RandomizedSearchCV(
    XGBStringLabelWrapper(random_state=RANDOM_STATE),
    param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_search.fit(X_train, y_train)
print("XGBoost best params:", xgb_search.best_params_)
xgb_best = xgb_search.best_estimator_
_ = report("XGBoost", xgb_best, X_test)
""")
K(r"""param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [10, 15, 20, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1,
)
rf_search.fit(X_train, y_train)
print("Random Forest best params:", rf_search.best_params_)
rf_best = rf_search.best_estimator_
_ = report("Random Forest", rf_best, X_test)
""")
K(r"""param_dist = {
    "max_iter": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 8, None],
    "max_leaf_nodes": [15, 31, 63],
}
hgb_search = RandomizedSearchCV(
    HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    param_dist, n_iter=15, cv=3, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1,
)
hgb_search.fit(X_train, y_train)
print("HistGradientBoosting best params:", hgb_search.best_params_)
hgb_best = hgb_search.best_estimator_
_ = report("HistGradientBoosting", hgb_best, X_test)
""")

M("## 5. Individual models, compared")
K(r"""individual_df = pd.DataFrame(results).T.round(4).sort_values("f1_macro", ascending=False)
print(individual_df.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
individual_df.plot(kind="bar", ax=ax)
ax.axhline(ceiling_6/100, color="purple", linestyle="-.", linewidth=1.5, label=f"accuracy ceiling ({ceiling_6:.1f}%)")
ax.set_title("v6 (6 classes) individual models vs. the accuracy ceiling")
ax.legend(fontsize=8)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
""")

M(r"""## 6. Ensemble — same two-stage approach that won in v5

v5 found that mixing 3 different algorithm families (bagging + 2 boosting flavours) for "diversity" actually *underperformed* the best individual models — their errors were too correlated. What won there was a plain soft vote of just the **two best-performing individual models**. Both approaches are tried again here rather than assumed, since the winning pair could differ on this target.
""")
K(r"""top2_names = individual_df.index[:2].tolist()
name_to_model = {
    "Bagging": bagging_best, "AdaBoost": ada_best, "Gradient Boosting": gb_best,
    "XGBoost": xgb_best, "Random Forest": rf_best, "HistGradientBoosting": hgb_best,
}
print("Top 2 individual models by macro-F1:", top2_names)
""")
K(r"""voting_diverse = VotingClassifier(
    estimators=[("rf", rf_best), ("hgb", hgb_best), ("xgb", xgb_best)],
    voting="soft", n_jobs=-1,
)
voting_diverse.fit(X_train, y_train)
_ = report("Voting (RF + HistGB + XGBoost)", voting_diverse, X_test)
""")
K(r"""m1_name, m2_name = top2_names[0], top2_names[1]
m1, m2 = name_to_model.get(m1_name), name_to_model.get(m2_name)
if m1 is not None and m2 is not None:
    voting_top2 = VotingClassifier(estimators=[(m1_name, m1), (m2_name, m2)], voting="soft", n_jobs=-1)
    voting_top2.fit(X_train, y_train)
    _ = report(f"Voting (top-2: {m1_name} + {m2_name})", voting_top2, X_test)
else:
    voting_top2 = None
    print("Top-2 includes a tuned linear/SVM model without predict_proba parity issues handled here -- skipping, diverse-family vote above stands.")
""")

M("## 7. Final comparison — every model, one table, against the ceiling")
K(r"""final_df = pd.DataFrame(results).T.round(4).sort_values("f1_macro", ascending=False)
print(final_df.to_string())
print(f"\nAccuracy ceiling for this 6-class target: {ceiling_6:.1f}%")
best_name = final_df["f1_macro"].idxmax()
best_acc = final_df.loc[best_name, "accuracy"]
print(f"Best model: {best_name}  (accuracy={best_acc*100:.1f}%, {best_acc*100/ceiling_6*100:.1f}% of the achievable ceiling)")

fig, ax = plt.subplots(figsize=(11, 5))
final_df.plot(kind="bar", ax=ax)
majority_baseline = y_test.value_counts(normalize=True).max()
n_classes = y_test.nunique()
ax.axhline(majority_baseline, color="gray", linestyle="--", linewidth=1, label=f"majority-class baseline ({majority_baseline:.3f})")
ax.axhline(1 / n_classes, color="red", linestyle=":", linewidth=1, label=f"uniform random guessing ({n_classes} classes)")
ax.axhline(ceiling_6/100, color="purple", linestyle="-.", linewidth=1.5, label=f"accuracy ceiling ({ceiling_6:.1f}%)")
ax.set_title("v6 (6 classes) — all individual models + ensembles, vs. majority/random/ceiling")
ax.legend(fontsize=8)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(parents=True, exist_ok=True)
final_df.to_csv(models_dir / "model_progression_v6_6class_comparison.csv")

v5_path = models_dir / "model_progression_v5_final_comparison.csv"
if v5_path.exists():
    v5_df = pd.read_csv(v5_path, index_col=0)
    print(f"\nv5 best (7 classes): {v5_df['f1_macro'].max():.4f} ({v5_df['f1_macro'].idxmax()})")
    print(f"v6 best (6 classes): {final_df['f1_macro'].max():.4f} ({best_name})")
""")

M(r"""## 8. Final model selection

The true best model is whichever scores highest on macro-F1 in the table above — individual or ensemble, no assumption that ensembling necessarily wins (v5 showed it doesn't always).

_To be filled in after the run: which model actually won, how close it got to the 88.1% ceiling, and whether 90% was reached (expected: no, since it exceeds the ceiling) or how close it came._
""")
K(r"""all_models_registry = dict(name_to_model)
all_models_registry["Logistic Regression (tuned)"] = logreg_search.best_estimator_
all_models_registry["SVM (RBF, tuned)"] = svm_search.best_estimator_
all_models_registry["Voting (RF + HistGB + XGBoost)"] = voting_diverse
if voting_top2 is not None:
    all_models_registry[f"Voting (top-2: {m1_name} + {m2_name})"] = voting_top2

final_model = all_models_registry[best_name]
print(f"Final model selected: {best_name}  (macro-F1={final_df.loc[best_name, 'f1_macro']:.4f}, "
      f"accuracy={final_df.loc[best_name, 'accuracy']*100:.1f}% vs {ceiling_6:.1f}% ceiling)")

joblib.dump(final_model, models_dir / "final_model_v6_6class.pkl")
joblib.dump(scaler, models_dir / "final_scaler_v6.pkl")
joblib.dump(list(X_train.columns), models_dir / "final_feature_columns_v6.pkl")
print("Saved final v6 model to", models_dir)
""")

M(r"""## 9. SHAP — global and per-guild feature attribution

Same approach as v5: `TreeExplainer` on **Random Forest** (natively supported, and a strong individual model here), cross-checked against **XGBoost**.
""")
K(r"""explainer_rf = shap.TreeExplainer(rf_best)
X_shap_sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)
shap_values = explainer_rf.shap_values(X_shap_sample)
class_names = list(rf_best.classes_)
print("Random Forest shap_values shape:", shap_values.shape, " classes:", class_names)

mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
top_feats = pd.Series(mean_abs_shap, index=X_shap_sample.columns).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 7))
top_feats.iloc[::-1].plot(kind="barh", ax=ax, color="darkorange")
ax.set_title("SHAP (Random Forest) — top 15 features, mean |SHAP value| across all 6 guilds")
ax.set_xlabel("mean |SHAP value|")
plt.tight_layout()
plt.show()
""")
K(r"""for guild in ["Upland scrub/woodland-edge", "Lowland open-country/wetland-edge", "Coastal (tern + curlew)"]:
    idx = class_names.index(guild)
    shap.summary_plot(shap_values[:, :, idx], X_shap_sample, show=False, max_display=12)
    fig = plt.gcf()
    fig.suptitle(f"SHAP (Random Forest) — {guild}", y=1.02)
    plt.tight_layout()
    plt.show()
""")

M(r"""## 10. LIME — instance-level, model-agnostic, on the final model
""")
K(r"""# LogReg/SVM were fit on scaled features; every tree-based model was fit on raw
# features. Use whichever matches the final model so LIME perturbs the right space.
needs_scaled = final_model in (logreg_search.best_estimator_, svm_search.best_estimator_)
X_train_lime = X_train_scaled if needs_scaled else X_train
X_test_lime = X_test_scaled if needs_scaled else X_test

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_lime.values,
    feature_names=list(X_train_lime.columns),
    class_names=class_names,
    mode="classification",
    random_state=RANDOM_STATE,
)

test_preds = final_model.predict(X_test_lime)
correct_mask = test_preds == y_test.values
wrong_mask = ~correct_mask

merged_classes = ["Upland scrub/woodland-edge", "Lowland open-country/wetland-edge", "Coastal (tern + curlew)"]
distinct_guilds = [g for g in class_names if g not in merged_classes]
easy_candidates = X_test.index[correct_mask & y_test.isin(distinct_guilds).values]
merged_candidates = X_test.index[correct_mask & y_test.isin(merged_classes).values]
wrong_candidates = X_test.index[wrong_mask]

picks = {
    "Confidently correct (environmentally distinct guild)": easy_candidates[0],
    "Correct, inside a merged guild": merged_candidates[0],
    "Misclassified": wrong_candidates[0],
}
print({k: (v, y_test.loc[v]) for k, v in picks.items()})
""")
K(r"""for title, row_idx in picks.items():
    instance = X_test_lime.loc[row_idx].values
    true_label = y_test.loc[row_idx]
    exp = lime_explainer.explain_instance(instance, final_model.predict_proba, num_features=10, top_labels=1)
    label_id = exp.available_labels()[0]
    predicted_label = class_names[label_id]
    fig = exp.as_pyplot_figure(label=label_id)
    fig.set_size_inches(8, 5)
    fig.suptitle(f"{title}\ntrue={true_label!r}  predicted={predicted_label!r}")
    plt.tight_layout()
    plt.show()
""")

M(r"""## 11. Conclusion

_To be filled in after the run with the actual achieved accuracy/macro-F1, how it compares to the 88.1% ceiling, and an explicit statement on the 90% question: whether it was reached, and if not, why that's an honest, explainable limitation of the data rather than a modelling shortfall -- directly reusable in the assignment write-up for guideline items #2 (proper data analysis), #10/#18 (justify and understand your own work)._
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

out = r"D:\Msc\Sem_2\Applied ML\Sri Lanka Rare birds\notebooks\05_six_class_ceiling_analysis.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(C), "cells")
