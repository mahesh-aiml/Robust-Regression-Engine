from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# All paths are relative to THIS file, so the app works from any folder
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "Advanced_Regression_HousePrice_Dataset_3800.xlsx"
RESULTS_PATH = BASE_DIR / "results.csv"
PLOT_PATH = BASE_DIR / "plots" / "08_model_compare.png"

FEATURES = ["area_sqft", "bedrooms", "bathrooms", "location_score", "property_age",
            "distance_city_km", "near_school", "near_metro", "crime_rate_index", "sale_year"]


def train_fallback():
    """Re-train the best model (SVR RBF) from the dataset if model.pkl is missing/incompatible."""
    df = pd.read_excel(DATA_PATH)
    df["sale_year"] = pd.to_datetime(df["sale_date"]).dt.year
    y = df["house_price_inr"] / 1e5
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], y, test_size=0.2, random_state=42)
    model = Pipeline([("sc", StandardScaler()),
                      ("m", SVR(kernel="rbf", C=300, gamma=0.01, epsilon=0.1))]).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    bundle = {"model": model, "features": FEATURES, "name": "SVR (RBF)",
              "test_r2": float(r2_score(y_te, pred)),
              "test_rmse": float(np.sqrt(mean_squared_error(y_te, pred)))}
    try:
        joblib.dump(bundle, MODEL_PATH)
    except Exception:
        pass
    return bundle


@st.cache_resource
def load():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        if DATA_PATH.exists():
            return train_fallback()
        raise


if not MODEL_PATH.exists() and not DATA_PATH.exists():
    st.error(f"Neither model.pkl nor the dataset .xlsx was found in:\n\n`{BASE_DIR}`\n\n"
             "Keep all project files together in one folder.")
    st.stop()

with st.spinner("Loading model..."):
    bundle = load()
model, features = bundle["model"], bundle["features"]

st.title("🏠 House Price Predictor")
st.caption(f"Model: {bundle['name']} | Test R²: {bundle['test_r2']:.3f} | "
           f"Test RMSE: {bundle['test_rmse']:.1f} lakh INR")

tab1, tab2 = st.tabs(["Predict", "Model comparison"])

with tab1:
    c1, c2 = st.columns(2)
    area = c1.number_input("Area (sqft)", 500, 4000, 1700)
    bed = c1.slider("Bedrooms", 1, 7, 3)
    bath = c1.slider("Bathrooms", 1, 6, 3)
    loc = c1.slider("Location score", 1.0, 10.0, 6.5, 0.1)
    age = c2.number_input("Property age (years)", 1, 80, 20)
    dist = c2.number_input("Distance to city (km)", 1.0, 40.0, 13.0, 0.1)
    crime = c2.slider("Crime rate index", 0.5, 12.0, 4.2, 0.1)
    year = c2.slider("Sale year", 2010, 2023, 2023)
    school = c1.checkbox("Near school", True)
    metro = c2.checkbox("Near metro", False)

    if st.button("Predict price", type="primary"):
        row = pd.DataFrame([{
            "area_sqft": area, "bedrooms": bed, "bathrooms": bath, "location_score": loc,
            "property_age": age, "distance_city_km": dist, "near_school": int(school),
            "near_metro": int(metro), "crime_rate_index": crime, "sale_year": year}])[features]
        lakh = float(model.predict(row)[0])
        st.success(f"Estimated price: ₹ {lakh * 1e5:,.0f}  ({lakh:,.1f} lakh)")
        st.info(f"Typical error is about ±{bundle['test_rmse']:.0f} lakh.")

with tab2:
    if RESULTS_PATH.exists():
        st.dataframe(pd.read_csv(RESULTS_PATH).set_index("Model"))
    else:
        st.warning("results.csv not found.")
    if PLOT_PATH.exists():
        st.image(str(PLOT_PATH))
