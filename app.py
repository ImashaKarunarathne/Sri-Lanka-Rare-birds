"""
Sri Lanka Bird Habitat-Guild Predictor -- Streamlit inference app.

Loads the v10 model (4 merged habitat guilds -- Coastal + Lowland open-country
+ Upland scrub merged into one "Open/Edge" class; trained in
notebooks/11_four_class_merged_model.ipynb, 94.2% accuracy / macro-F1 0.9421,
94.7% of the 99.5% accuracy ceiling) and lets a user enter environmental/
climate readings for an observation site to predict which bird habitat guild
is most likely to be present there.

Run with:  streamlit run app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(PROJECT_ROOT))
from model_utils import XGBStringLabelWrapper  # noqa: F401 -- harmless if unused; the v10 model is a plain BaggingClassifier with no XGBoost component, but this keeps the loader compatible with the v9 stacking model's artifacts too

st.set_page_config(page_title="Sri Lanka Bird Habitat-Guild Predictor", page_icon="🦜", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "final_model_v10_stacking.pkl")
    scaler = joblib.load(MODELS_DIR / "final_scaler_v10.pkl")
    feature_columns = joblib.load(MODELS_DIR / "final_feature_columns_v10.pkl")
    numeric_cols = joblib.load(MODELS_DIR / "final_numeric_cols_v10.pkl")
    categorical_cols = joblib.load(MODELS_DIR / "final_categorical_cols_v10.pkl")
    return model, scaler, feature_columns, numeric_cols, categorical_cols


model, scaler, FEATURE_COLUMNS, NUMERIC_COLS, CATEGORICAL_COLS = load_artifacts()

PROVINCES = [
    "Ampara", "Anuradhapura", "Badulla", "Batticaloa", "Colombo", "Galle", "Gampaha",
    "Hambantota", "Jaffna", "Kalutara", "Kandy", "Kegalla", "Kilinochchi", "Kurunegala",
    "Mannar", "Matale", "Matara", "Monaragala", "Mullaittivu", "Nuwara", "Polonnaruwa",
    "Puttalam", "Ratnapura", "Trincomalee", "Vavuniya",
]
LANDCOVER_CLASSES = [2, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17]

st.title("🦜 Sri Lanka Bird Habitat-Guild Predictor")
st.caption(
    "Enter the environmental and climate readings for an observation site to predict "
    "which of 4 bird habitat guilds is most likely present (Rainforest, Highland, "
    "Urban/wetland-edge, or the merged Open/Edge lowland-coastal-scrub class). Trained "
    "on the Sri Lanka Bird Diversity Dataset (GBIF occurrences + remote-sensing/climate "
    "features)."
)

with st.form("input_form"):
    st.subheader("Location")
    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input("Latitude", min_value=5.9, max_value=9.9, value=7.0, step=0.01, format="%.4f")
        province = st.selectbox("Province", PROVINCES, index=PROVINCES.index("Colombo"))
        elevation = st.number_input("Elevation (m)", min_value=-10, max_value=2500, value=50, step=10)
    with c2:
        lon = st.number_input("Longitude", min_value=79.6, max_value=82.0, value=80.0, step=0.01, format="%.4f")
        landcover = st.selectbox("Land cover class code", LANDCOVER_CLASSES, index=LANDCOVER_CLASSES.index(13))
        individual_count = st.number_input("Flock / group size seen (individualCount)", min_value=1, max_value=500, value=2, step=1)

    st.subheader("Vegetation & climate")
    c3, c4 = st.columns(2)
    with c3:
        ndvi = st.slider("NDVI (vegetation index)", -0.2, 1.0, 0.55, 0.01)
        temp_mean = st.slider("Mean temperature (°C)", 18.0, 33.0, 27.0, 0.1)
        rainfall = st.slider("Rainfall (mm)", 0.0, 100.0, 5.0, 0.5)
        wind_mean = st.slider("Mean wind speed", 0.0, 10.0, 3.0, 0.1)
    with c4:
        humid_mean = st.slider("Mean humidity (%)", 50.0, 97.0, 80.0, 0.5)
        shortwave_radiation = st.slider("Shortwave radiation", 0.0, 28.0, 18.0, 0.5)
        avg_rad = st.slider("Average surface radiation (avg_rad)", -1.0, 51.0, 2.0, 0.5)
        lka_general_2020 = st.slider("Sri Lanka environmental baseline index", 0.0, 5200.0, 10.0, 1.0)

    with st.expander("Aerosol columns (advanced -- defaults are typical values)"):
        carbon_mass = st.number_input("Carbon_Mass", value=5e-10, format="%.2e")
        dust_mass = st.number_input("Dust_Mass", value=5e-9, format="%.2e")
        so2_mass = st.number_input("SO2_Mass", value=1e-9, format="%.2e")
        sulfate_mass = st.number_input("Sulfate_Mass", value=2e-9, format="%.2e")
        sea_salt_mass = st.number_input("Sea_Salt_Mass", value=3e-8, format="%.2e")
        total_aerosol_extinction = st.number_input("Total_Aerosol_Extinction", value=0.18, format="%.4f")

    submitted = st.form_submit_button("Predict habitat guild")

if submitted:
    raw_row = pd.DataFrame([{
        "decimalLatitude": lat, "decimalLongitude": lon, "avg_rad": avg_rad, "NDVI": ndvi,
        "elevation_meters": elevation, "Carbon_Mass": carbon_mass, "Dust_Mass": dust_mass,
        "SO2_Mass": so2_mass, "Sulfate_Mass": sulfate_mass, "Sea_Salt_Mass": sea_salt_mass,
        "Total_Aerosol_Extinction": total_aerosol_extinction, "temp_mean": temp_mean,
        "rainfall": rainfall, "wind_mean": wind_mean, "humid_mean": humid_mean,
        "shortwave_radiation": shortwave_radiation, "lka_general_2020": lka_general_2020,
        "individualCount": individual_count, "stateProvince": province, "LandCover_Class": landcover,
    }])

    X_new = pd.get_dummies(raw_row, columns=CATEGORICAL_COLS, drop_first=True)
    X_new = X_new.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]
    proba_df = pd.DataFrame({"guild": model.classes_, "probability": proba}).sort_values(
        "probability", ascending=False
    )

    st.success(f"Predicted habitat guild: **{pred}**")
    st.bar_chart(proba_df.set_index("guild")["probability"])
    st.dataframe(proba_df.reset_index(drop=True), use_container_width=True)

st.divider()
st.caption(
    "Model: tuned Bagging ensemble over decision trees, 94.2% test accuracy / macro-F1 "
    "0.9421 (94.7% of the measured 99.5% accuracy ceiling) "
    "(see notebooks/11_four_class_merged_model.ipynb for training, validation, "
    "SHAP/LIME interpretation, and the duplicate-row accuracy-ceiling analysis this "
    "result is benchmarked against)."
)
