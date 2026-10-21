import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

@st.cache_resource
def load():
    return joblib.load("model.pkl")

bundle = load()
model, features, ranges = bundle["model"], bundle["features"], bundle["ranges"]

st.title("🏠 House Price Predictor")
st.caption(f"Model: {bundle['name']} | Test R²: {bundle['test_r2']:.3f} | Test RMSE: {bundle['test_rmse']:.1f} lakh INR")

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
    st.dataframe(pd.read_csv("results.csv").set_index("Model"))
    st.image("plots/08_model_compare.png")
