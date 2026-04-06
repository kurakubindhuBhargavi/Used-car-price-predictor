import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import pickle

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Dataframe text */
[data-testid="stDataFrame"] {
    color: white;
}

/* Table header */
thead tr th {
    color: white !important;
    background-color: rgba(255,255,255,0.1) !important;
}

/* Table cells */
tbody tr td {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
    <div style="
        background: linear-gradient(90deg, #ff512f, #dd2476);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);">
        <h1 style="color:white;">🚗 Used Car Price Predictor</h1>
        <p style="color:white; font-size:18px;">💡 Smart AI Prediction System</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #232526, #414345);
    color: white;
}

/* Sidebar title */
section[data-testid="stSidebar"] h2 {
    color: #00e676;
    text-align: center;
}

/* Labels */
label {
    color: #ffffff !important;
    font-weight: 500;
}

/* Input fields */
div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.9) !important;
    border-radius: 8px;
}

/* Number input */
input {
    background-color: rgba(255,255,255,0.9) !important;
    border-radius: 8px;
}

/* Radio buttons spacing */
.stRadio > div {
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* All headings */
h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

/* Paragraph text */
p {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
st.write("\n\n"*2)

filename = 'Auto_Price_Pred_Model.pkl'
model, model_cols = pickle.load(open(filename, 'rb'))
with st.sidebar:
    st.markdown("### 🚘 Basic Info")

    make_model = st.selectbox(
        "🚘 Model Selection",
        ("Audi A3", "Audi A1", "Opel Insignia", "Opel Astra", "Opel Corsa", "Renault Clio", "Renault Espace", "Renault Duster")
    )

    hp_kW = st.number_input("🔥 Horse Power", min_value=40, max_value=294, value=120, step=5)

    age = st.number_input("📅 Age (years)", min_value=0, max_value=10, value=0, step=1)

    km = st.number_input("📍 KM Driven", min_value=0, max_value=317000, value=10000, step=5000)

    Gears = st.number_input("⚙️ Gears", min_value=5, max_value=8, value=5, step=1)

    Gearing_Type = st.radio("🔧 Gearing Type", ("Manual", "Automatic", "Semi-automatic"))

    fuel = st.selectbox("⛽ Fuel Type", ("Petrol", "Diesel", "Hybrid"))

    body_type = st.selectbox("🚙 Body Type", ("Sedan", "Hatchback", "SUV"))

    previous_owners = st.number_input("👤 Previous Owners", min_value=0, max_value=5, value=1)

    displacement = st.number_input("🛠️ Engine CC", min_value=800, max_value=5000, value=1500)

    weight = st.number_input("🏋️ Weight (kg)", min_value=800, max_value=3000, value=1200)

    drive_chain = st.selectbox("🚀 Drive Type", ("FWD", "RWD", "AWD"))

my_dict = {
    "make_model": make_model,
    "hp_kW": hp_kW,
    "age": age,
    "km": km,
    "Gears": Gears,
    "Gearing_Type": Gearing_Type,
    "Fuel": fuel,
    "body_type": body_type,
    "Previous_Owners": previous_owners,
    "Displacement_cc": displacement,
    "Weight_kg": weight,
    "Drive_chain": drive_chain
}
df = pd.DataFrame([my_dict])
df_show = df.copy()
cols = {
    "make_model": "Car Model",
    "hp_kW": "Horse Power",
    "age": "Age",
    "km": "KM Traveled",
    "Gears": "Gears",
    "Gearing_Type": "Gearing Type",
    "Fuel": "Fuel",
    "body_type": "Body Type",
    "Previous_Owners": "Owners",
    "Displacement_cc": "Engine CC",
    "Weight_kg": "Weight (kg)",
    "Drive_chain": "Drive Type"
}
df_show.rename(columns=cols, inplace=True)

# show table

st.markdown("## 📊 <span style='color:white;'>Selected Car Specifications</span>", unsafe_allow_html=True)
st.dataframe(df_show)


cols = {
    "make_model": "Car Model",
    "hp_kW": "Horse Power",
    "age": "Age",
    "km": "km Traveled",
    "Gears": "Gears",
    "Gearing_Type": "Gearing Type",
    "Fuel": "Fuel",
    "body_type": "Body Type",
    "Previous_Owners": "Owners",
    "Displacement_cc": "Engine CC",
    "Weight_kg": "Weight",
    "Drive_chain": "Drive Type"
}

df_show.rename(columns=cols, inplace=True)
st.dataframe(df_show.style.set_properties(**{
    'background-color': '#f0f2f6',
    'color': 'black',
    'border-color': 'white'
}))
df = pd.DataFrame.from_dict([my_dict])

cols = {
    "make_model": "Car Model",
    "hp_kW": "Horse Power",
    "age": "Age",
    "km": "km Traveled",
    "Gears": "Gears",
    "Gearing_Type": "Gearing Type"
}

df_show = df.copy()
df_show.rename(columns = cols, inplace = True)


if st.button("🚀 Predict Price"):

    df = pd.get_dummies(df)
    df = df.reindex(columns=model_cols, fill_value=0)

    pred = model.predict(df)

    price_inr = pred[0] * 90   # conversion

    st.markdown("## 💰 <span style='color:white;'>Prediction Result</span>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #00c853, #64dd17);
        padding: 25px;
        border-radius: 15px;
        text-align: center;">
        <h1 style="color:black;">₹ {int(price_inr):,}</h1>
        <p style="color:black;">Estimated Car Price (INR)</p>
    </div>
    """, unsafe_allow_html=True)
    # INR BIG PREMIUM BOX
    

    # Extra info
    st.info("📊 Prediction based on selected car features using Machine Learning model")

st.write("\n\n")

