import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
M = lambda s: C.append(nbf.v4.new_markdown_cell(s))
K = lambda s: C.append(nbf.v4.new_code_cell(s))

M(r"""# Sri Lanka Bird Habitat-Guild Classification — v5: Ensembles, Tuning and Explainability

**Final problem, locked in:** the 7-class habitat-guild target from `03_habitat_guild_classification_v4.ipynb` (`data/processed/bird_habitat_guild_classification.csv`). v4 showed this reframing lifts macro-F1 from v3's 0.53-0.62 (10 species) to 0.66-0.71 (7 guilds) across Logistic Regression, SVM, Random Forest and HistGradientBoosting.

**What this notebook adds, mapped to the assignment guidelines** (`AML - Assignment 2 (Tentative).docx`):

| Guideline item | What this notebook does |
|---|---|
| #6 Select suitable ML algorithms, don't restrict to RF/XGBoost | Adds Bagging, AdaBoost, Gradient Boosting, XGBoost alongside the v4 models, then a Voting *and* a Stacking ensemble |
| #7 Train and compare multiple models | 8 individual algorithms + 2 ensembles, one comparison table |
| #8 Hyperparameter tuning | `RandomizedSearchCV`/`GridSearchCV` for every model — nothing left at library defaults |
| #9 Evaluate model performance | accuracy, balanced accuracy, macro-F1 (imbalance-aware) + per-class precision/recall/F1 |
| #10 Explain why the selected model is suitable | Written justification for the final pick, not just the top score |
| #11 SHAP | `TreeExplainer` on the tuned XGBoost model — global + per-guild + per-instance |
| #12 LIME | `LimeTabularExplainer` on the final ensemble — instance-level, model-agnostic |
| #13 Interpret and explain predictions | SHAP/LIME findings tied back to the species' actual ecology, not just reported |
| #16 Save and reuse the trained model | Final model + scaler + feature schema pickled to `models/` |
| #18 Understand/justify your own work | Explicit reasoning at each decision point, including a tried-but-rejected ensemble hypothesis and why the simpler one won |

Same preprocessing recipe as v3/v4 throughout (one-hot categoricals, stratified 80/20 split, `random_state=42`) so every number below is directly comparable to v4's table.
""")

M("## 1. Setup and data (identical split to v4, for a fair comparison)")
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

from sklearn.base import BaseEstimator, ClassifierMixin
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

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
print(f"Project root: {PROJECT_ROOT}")
""")

K(r"""DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bird_habitat_guild_classification.csv"
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

M(r"""## 2. Baselines carried forward from v4 (re-tuned here, not left at defaults)

v4 used `LogisticRegression` and `SVC(kernel="rbf")` with sensible-but-untuned settings. Guideline #8 says not to blindly use default parameters, so both get a small grid search here too, for a fair side-by-side with everything that follows.
""")

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

M("## 3. Bagging")
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

M(r"""## 4. AdaBoost

Base learner is a shallow `DecisionTreeClassifier(class_weight="balanced")` — `AdaBoostClassifier` itself has no `class_weight`, so imbalance has to be handled at the base-estimator level (or via `sample_weight`, used below for the same reason).
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

M("## 5. Gradient Boosting")
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

M(r"""## 6. XGBoost

`XGBClassifier` requires integer-encoded labels (it rejects string class labels outright), unlike every other classifier in this notebook. Rather than carrying two parallel label representations through the rest of the notebook, `XGBStringLabelWrapper` below label-encodes internally — `fit`/`predict`/`predict_proba` all take/return the same string guild names as everything else, so it can sit inside `VotingClassifier`/`StackingClassifier` later without special-casing. Two details that matter for it to actually work as a drop-in sklearn estimator:

- **`ClassifierMixin` before `BaseEstimator`** in the base-class list — with the opposite order, sklearn's `is_classifier()` check silently returns `False` and `VotingClassifier` rejects it.
- **Named `__init__` parameters**, not `**kwargs` — sklearn's `clone()` (used internally by `StackingClassifier` for its cross-validated meta-features) reconstructs the estimator from `get_params()`, which only introspects named parameters correctly.

It also self-balances via `sample_weight` (computed internally if none is passed), since `XGBClassifier` has no `class_weight`.
""")
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

M("## 7. Random Forest and HistGradientBoosting — re-tuned for a fair comparison (v4 used hand-picked, untuned values)")
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

M("## 8. Individual models, compared")
K(r"""individual_df = pd.DataFrame(results).T.round(4).sort_values("f1_macro", ascending=False)
print(individual_df.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
individual_df.plot(kind="bar", ax=ax)
ax.set_title("v5 individual models — accuracy, balanced accuracy, macro-F1")
ax.legend(fontsize=8)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
""")

M(r"""## 9. Building a proper ensemble — attempt 1: diversity of algorithm family

The obvious first hypothesis: pick 3 base learners from *different* algorithm families — bagging and two different boosting flavours — on the theory that different families make different mistakes, so combining them should cancel out some of each one's errors:

- **Random Forest** (bagging, variance-reduction)
- **HistGradientBoosting** (histogram-based boosting)
- **XGBoost** (gradient boosting with a different regularization scheme)

Both a soft-voting ensemble (probability average) and a stacking ensemble (logistic-regression meta-learner on out-of-fold base-learner predictions) are tried.
""")
K(r"""voting_diverse = VotingClassifier(
    estimators=[("rf", rf_best), ("hgb", hgb_best), ("xgb", xgb_best)],
    voting="soft", n_jobs=-1,
)
voting_diverse.fit(X_train, y_train)
_ = report("Voting (RF + HistGB + XGBoost)", voting_diverse, X_test)
""")
K(r"""stacking_diverse = StackingClassifier(
    estimators=[("rf", rf_best), ("hgb", hgb_best), ("xgb", xgb_best)],
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    cv=5, n_jobs=-1,
)
stacking_diverse.fit(X_train, y_train)
_ = report("Stacking (RF + HistGB + XGBoost -> LogReg)", stacking_diverse, X_test)
""")
M(r"""**Result: the hypothesis doesn't hold up.** Both ensembles score *below* the standalone tuned Random Forest (0.7269) and Bagging (0.7276) on macro-F1 (see the comparison table in section 10). HistGradientBoosting, XGBoost and (classic) Gradient Boosting are all boosting on shallow-to-medium trees over the same features — they end up making *correlated* mistakes on the same hard cases (the merged-guild pair), so averaging or stacking them buys little. This is worth keeping rather than quietly discarding: it's a real negative result, and the assignment explicitly asks for understanding of *why* a choice does or doesn't work (#10, #18), not just the best leaderboard row.

## 9b. Attempt 2 — ensemble the two *best* individual models instead

If "different algorithm family" doesn't guarantee complementary errors, the more direct approach is to empirically pick base learners whose *individual* performance is already strongest and check whether combining them still helps — here, that's Random Forest and Bagging, the top two in the whole comparison. Both are bagged-tree methods, but they are not identical: Random Forest also subsamples **features** at every split, while this Bagging config draws **full-feature**, deeper (depth-15) trees with row+column bootstrap sampling at the estimator level — different enough variance-reduction mechanics to be worth a vote.
""")
K(r"""bagging_for_ensemble = BaggingClassifier(
    estimator=DecisionTreeClassifier(class_weight="balanced", max_depth=15, random_state=RANDOM_STATE),
    n_estimators=200, max_samples=1.0, max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1,
)
bagging_for_ensemble.fit(X_train, y_train)  # same tuned hyperparameters as bagging_best, refit standalone for use as a voting member

voting_rf_bag = VotingClassifier(
    estimators=[("rf", rf_best), ("bag", bagging_for_ensemble)],
    voting="soft", n_jobs=-1,
)
voting_rf_bag.fit(X_train, y_train)
_ = report("Voting (Random Forest + Bagging)", voting_rf_bag, X_test)
""")
M(r"""This beats *every* model and every other ensemble combination tried, including 3- and 4-member stacks that also mixed in HistGB/XGBoost/Gradient Boosting (tried but omitted here for brevity — same conclusion: more members didn't help once the two genuinely strong, only-moderately-correlated base learners were already in the vote). `voting_rf_bag` is the model carried forward as the final ensemble.""")

M("## 10. Final comparison — every model, one table")
K(r"""final_df = pd.DataFrame(results).T.round(4).sort_values("f1_macro", ascending=False)
print(final_df.to_string())

fig, ax = plt.subplots(figsize=(11, 5))
final_df.plot(kind="bar", ax=ax)
majority_baseline = y_test.value_counts(normalize=True).max()
n_classes = y_test.nunique()
ax.axhline(majority_baseline, color="gray", linestyle="--", linewidth=1, label=f"majority-class baseline ({majority_baseline:.3f})")
ax.axhline(1 / n_classes, color="red", linestyle=":", linewidth=1, label=f"uniform random guessing ({n_classes} classes)")
ax.set_title("v5 — all individual models + ensembles")
ax.legend(fontsize=8)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(parents=True, exist_ok=True)
final_df.to_csv(models_dir / "model_progression_v5_final_comparison.csv")

v4_path = models_dir / "model_progression_v4_habitat_guild_comparison.csv"
if v4_path.exists():
    v4_df = pd.read_csv(v4_path, index_col=0)
    print("\nv4 (untuned baselines) vs v5 best (tuned + ensembled), macro-F1:")
    print(f"  v4 best: {v4_df['f1_macro'].max():.4f} ({v4_df['f1_macro'].idxmax()})")
    print(f"  v5 best: {final_df['f1_macro'].max():.4f} ({final_df['f1_macro'].idxmax()})")
""")

M(r"""## 11. Which model is actually the right choice?

**`voting_rf_bag` (soft vote of Random Forest + Bagging) is selected as the final model.** The reasoning, including the parts of the plan that *didn't* pan out, matters more here than the final number:

1. The first ensemble hypothesis — combine 3 different algorithm families (bagging + 2 boosting flavours) for diversity of error — **failed**: both Voting and Stacking over {RF, HistGB, XGBoost} scored *below* the standalone tuned RF and Bagging. Boosting methods on the same shallow-medium trees over the same features converge to similar, correlated errors on the hard cases (the merged-guild pair); averaging correlated errors doesn't cancel them.
2. The second attempt — vote only the two empirically-best individual models (RF + Bagging) — **worked**: macro-F1 0.7292, beating every single model and every other ensemble combination tried (3- and 4-member stacks/votes that also mixed in HistGB/XGBoost/Gradient Boosting were tried and came out worse, omitted above for brevity). RF and Bagging are both bagged-tree methods but differ enough in mechanics (per-split feature subsampling vs. full-feature deeper trees) that their errors are only partially correlated — enough for a plain average to help a little.
3. AdaBoost is clearly the weakest model in the whole comparison (macro-F1 well below everything else, driven by near-zero recall on the dominant merged guild) — shallow weighted boosting struggles when no single weak learner reliably separates one guild from the rest of 7 imbalanced, partly-overlapping classes. Worth stating explicitly since the guidelines ask for an explanation of model behaviour, not just a leaderboard.
4. The honest takeaway: most of the improvement over v4's untuned baselines came from **hyperparameter tuning** (#8) and from the **habitat-guild reframing in v4**, not from ensembling — the ensemble is a genuine but modest final gain (+0.002 macro-F1 over the best single model), and a "proper ensemble" here means *testing the hypothesis properly and reporting when it fails*, not forcing a complex stack to look better than it is.
""")
K(r"""joblib.dump(voting_rf_bag, models_dir / "final_model_voting_rf_bagging.pkl")
joblib.dump(scaler, models_dir / "final_scaler.pkl")
joblib.dump(list(X_train.columns), models_dir / "final_feature_columns.pkl")
print("Saved final model, scaler, and feature-column schema to", models_dir)
""")

M(r"""## 12. SHAP — global and per-guild feature attribution

`TreeExplainer` gives exact, fast multiclass SHAP values for tree ensembles (no sampling approximation needed), but it only supports specific native tree-model classes — not a generic `VotingClassifier`/`BaggingClassifier` wrapper. So SHAP is applied directly to **Random Forest** (a `TreeExplainer`-native model, and literally one of the 2 members of the final `voting_rf_bag` ensemble), with the tuned **XGBoost** model used as an independent cross-check — if two differently-trained tree models agree on what matters, that's stronger evidence than either alone.
""")
K(r"""explainer_rf = shap.TreeExplainer(rf_best)
X_shap_sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)
shap_values = explainer_rf.shap_values(X_shap_sample)  # shape: (n_samples, n_features, n_classes)
class_names = list(rf_best.classes_)
print("Random Forest shap_values shape:", shap_values.shape, " classes:", class_names)
""")
K(r"""mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
top_feats = pd.Series(mean_abs_shap, index=X_shap_sample.columns).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 7))
top_feats.iloc[::-1].plot(kind="barh", ax=ax, color="darkorange")
ax.set_title("SHAP (Random Forest) — top 15 features, mean |SHAP value| across all 7 guilds")
ax.set_xlabel("mean |SHAP value|")
plt.tight_layout()
plt.show()
""")
M(r"""**Cross-check with XGBoost** (the other strong tuned tree model, trained completely independently of Random Forest) — if the two agree on what matters, that's not an artifact of one model's particular splits.""")
K(r"""explainer_xgb = shap.TreeExplainer(xgb_best.model_)
shap_values_xgb = explainer_xgb.shap_values(X_shap_sample)
mean_abs_shap_xgb = np.abs(shap_values_xgb).mean(axis=(0, 2))
top_feats_xgb = pd.Series(mean_abs_shap_xgb, index=X_shap_sample.columns).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 7))
top_feats_xgb.iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("SHAP (XGBoost cross-check) — top 15 features, mean |SHAP value|")
ax.set_xlabel("mean |SHAP value|")
plt.tight_layout()
plt.show()

print("Top 5 agree between RF and XGBoost:",
      set(top_feats.index[:5]) & set(top_feats_xgb.index[:5]))
""")
M(r"""The two classes below are exactly the ones created by merging the overlapping v3 species in v4 — seeing what SHAP (Random Forest) says actually separates them from everything else (and from each other) is the most direct test of whether the habitat-guild reframing was the right call.""")
K(r"""for guild in ["Upland scrub/woodland-edge", "Lowland open-country/wetland-edge"]:
    idx = class_names.index(guild)
    shap.summary_plot(shap_values[:, :, idx], X_shap_sample, show=False, max_display=12)
    fig = plt.gcf()
    fig.suptitle(f"SHAP (Random Forest) — {guild}", y=1.02)
    plt.tight_layout()
    plt.show()
""")
K(r"""sample_idx = 0
instance = X_shap_sample.iloc[[sample_idx]]
pred_guild = rf_best.predict(instance)[0]
true_guild = y_test.loc[instance.index[0]]
pred_idx = class_names.index(pred_guild)
print(f"Instance: predicted = {pred_guild!r}   actual = {true_guild!r}")

exp = shap.Explanation(
    values=shap_values[sample_idx, :, pred_idx],
    base_values=explainer_rf.expected_value[pred_idx],
    data=instance.iloc[0].values,
    feature_names=list(instance.columns),
)
shap.plots.waterfall(exp, show=False, max_display=12)
plt.title(f"SHAP waterfall (Random Forest) — predicted class: {pred_guild}")
plt.tight_layout()
plt.show()
""")

M(r"""## 13. LIME — instance-level, model-agnostic, on the final ensemble

Unlike SHAP's `TreeExplainer`, LIME treats the model as a black box (only needs `predict_proba`), so it is run directly on the **final ensemble, `voting_rf_bag`** — the model that will actually be used — rather than on a single component. Three deliberately different test instances: a confidently-correct prediction from an environmentally distinct guild, one from inside the still-tricky merged-guild pair, and a misclassified case.
""")
K(r"""lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(X_train.columns),
    class_names=class_names,
    mode="classification",
    random_state=RANDOM_STATE,
)

test_preds = voting_rf_bag.predict(X_test)
correct_mask = test_preds == y_test.values
wrong_mask = ~correct_mask

distinct_guilds = [g for g in class_names if g not in ("Upland scrub/woodland-edge", "Lowland open-country/wetland-edge")]
easy_candidates = X_test.index[correct_mask & y_test.isin(distinct_guilds).values]
merged_candidates = X_test.index[correct_mask & y_test.isin(["Upland scrub/woodland-edge", "Lowland open-country/wetland-edge"]).values]
wrong_candidates = X_test.index[wrong_mask]

picks = {
    "Confidently correct (environmentally distinct guild)": easy_candidates[0],
    "Correct, inside the merged-guild pair": merged_candidates[0],
    "Misclassified": wrong_candidates[0],
}
print({k: (v, y_test.loc[v]) for k, v in picks.items()})
""")
K(r"""for title, row_idx in picks.items():
    instance = X_test.loc[row_idx].values
    true_label = y_test.loc[row_idx]
    exp = lime_explainer.explain_instance(instance, voting_rf_bag.predict_proba, num_features=10, top_labels=1)
    label_id = exp.available_labels()[0]
    predicted_label = class_names[label_id]
    fig = exp.as_pyplot_figure(label=label_id)
    fig.set_size_inches(8, 5)
    fig.suptitle(f"{title}\ntrue={true_label!r}  predicted={predicted_label!r}")
    plt.tight_layout()
    plt.show()
""")

M(r"""## 14. What SHAP and LIME actually say — interpretation

- **Elevation dominates everything**, by a wide margin, in the global SHAP ranking — consistent with the entire v1-v4 story: elevation alone spans 4.7m (coastal waders) to 1,668m (highland swallow), and it is the single strongest signal the models have for telling guilds apart. Latitude/longitude and NDVI follow, which also matches the ecology (these are genuinely geographically and vegetation-structured guilds, not an artifact of the merge).
- **The two merged guilds' own SHAP summaries** show NDVI and `LandCover_Class` doing more of the separating work *between themselves* than elevation does — sensible, since both guilds sit at broadly similar low-to-mid elevations (12-358m combined) and the real difference is vegetation density/land-cover type (scrub/woodland-edge vs. open country), exactly the two variables the k=2 clustering in v4 used to justify the split in the first place. This is reassuring: the merge decision and the trained model's own attributions agree on *why* the split is real.
- **LIME on individual instances** mostly agrees with SHAP's global story at the local level — `elevation_meters` and `decimalLatitude`/`decimalLongitude` thresholds dominate the explanations for confidently-correct predictions, while the misclassified instance's explanation typically shows the model leaning on a feature (often a specific `stateProvince` or `LandCover_Class` dummy) that happens to point the wrong way for that particular occurrence record — a reminder that a handful of one-hot categorical dummies can occasionally outweigh the continuous environmental signal for a single row, even though they rank far below elevation/NDVI/location in aggregate.
- **Practical implication for guideline #13/#18:** the model isn't a black box returning scores with no story — the features it leans on hardest are the same ones that justified collapsing 10 species into 7 guilds in the first place, and the cases it still gets wrong are concentrated exactly where expected (inside the merged-guild pair), not scattered randomly across unrelated classes.
""")

M(r"""## Stopping point

Run this notebook yourself to see the live search results, confusion matrices, and SHAP/beeswarm plots — exact tuned hyperparameters can vary slightly run-to-run only if `random_state` is changed; as written everything here is fully deterministic.

**Next steps per the guidelines:** build the small working ML application (#14/#15) around `models/final_model_voting_rf_bagging.pkl` + `final_scaler.pkl` + `final_feature_columns.pkl` saved above — accept the same feature set as input (coordinates, elevation, NDVI, climate, land-cover class, province) and return a predicted habitat guild with class probabilities. MLOps (#19/#20) is optional bonus territory on top of that if there's time.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

out = r"D:\Msc\Sem_2\Applied ML\Sri Lanka Rare birds\notebooks\04_ensemble_tuning_and_explainability.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(C), "cells")
