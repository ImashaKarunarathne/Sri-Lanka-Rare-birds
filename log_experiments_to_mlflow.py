"""
Logs the model comparison results from notebooks 10 and 11 to DagsHub's hosted
MLflow tracking server, so every model that was trained and tuned shows up as
a comparable experiment run (metrics + params) in the DagsHub "Experiments" tab.

Requires: `dagshub login` already completed once on this machine (stores an
OAuth token locally; this script never touches the raw token).

Run with:  python log_experiments_to_mlflow.py
"""
import joblib
import mlflow
import pandas as pd
import dagshub

REPO_OWNER = "ImashaKarunarathne"
REPO_NAME = "Sri-Lanka-Rare-birds"

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

RUNS = [
    {
        "csv": "models/model_progression_v9_stacking_comparison.csv",
        "notebook": "10_stacking_ensemble_final_model.ipynb",
        "dataset_version": "v9_6class",
        "n_classes": 6,
        "accuracy_ceiling": 0.979,
    },
    {
        "csv": "models/model_progression_v10_stacking_comparison.csv",
        "notebook": "11_four_class_merged_model.ipynb",
        "dataset_version": "v10_4class_merged",
        "n_classes": 4,
        "accuracy_ceiling": 0.995,
    },
]

mlflow.set_experiment("bird-habitat-guild-classification")

for run_cfg in RUNS:
    df = pd.read_csv(run_cfg["csv"], index_col=0)
    for model_name, row in df.iterrows():
        with mlflow.start_run(run_name=f"{run_cfg['dataset_version']}__{model_name}"):
            mlflow.set_tags(
                {
                    "notebook": run_cfg["notebook"],
                    "dataset_version": run_cfg["dataset_version"],
                    "model_family": model_name,
                }
            )
            mlflow.log_params(
                {
                    "n_classes": run_cfg["n_classes"],
                    "accuracy_ceiling": run_cfg["accuracy_ceiling"],
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": row["accuracy"],
                    "balanced_accuracy": row["balanced_accuracy"],
                    "f1_macro": row["f1_macro"],
                    "ceiling_utilization": row["accuracy"] / run_cfg["accuracy_ceiling"],
                }
            )
    print(f"Logged {len(df)} runs for {run_cfg['dataset_version']}")

# Also log the final selected model (v10 Bagging) with its artifacts, as the
# "production" run that app.py actually serves.
with mlflow.start_run(run_name="v10_4class_merged__FINAL_Bagging"):
    mlflow.set_tags(
        {
            "notebook": "11_four_class_merged_model.ipynb",
            "dataset_version": "v10_4class_merged",
            "model_family": "Bagging",
            "stage": "production",
        }
    )
    mlflow.log_params({"n_classes": 4, "accuracy_ceiling": 0.995})
    mlflow.log_metrics(
        {
            "accuracy": 0.9422,
            "balanced_accuracy": 0.9422,
            "f1_macro": 0.9421,
            "ceiling_utilization": 0.9422 / 0.995,
        }
    )
    model = joblib.load("models/final_model_v10_stacking.pkl")
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    mlflow.log_artifact("models/final_scaler_v10.pkl")
    mlflow.log_artifact("models/final_feature_columns_v10.pkl")

print("Done. View runs at: https://dagshub.com/%s/%s.mlflow" % (REPO_OWNER, REPO_NAME))
