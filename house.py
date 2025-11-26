import pickle
import streamlit as st
import numpy as np

# Load data
with open("house.pkl", "rb") as file:
    loaded_data = pickle.load(file)

# Page config
st.set_page_config(page_title="House Price Prediction", layout="centered")

# Custom CSS for futuristic neon look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap');
    body {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #0a0e17, #1a1f2e);
        color: #00ffcc;
    }
    .stApp {
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 0 40px rgba(0, 255, 204, 0.3);
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
        color: #000000 !important;
        font-size: 18px;
        font-weight: 500;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        border: 2px solid #00ffcc;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px #ff00ff;
        transform: scale(1.05);
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(0, 255, 204, 0.1);
        border-radius: 8px;
        border: 1px solid #00ffcc;
        color: #00ffcc !important;
    }
    .stSelectbox div[data-baseweb="popover"] {
        background-color: #0a0e17;
        color: #00ffcc !important;
    }
    .stNumberInput input {
        background-color: rgba(0, 255, 204, 0.1);
        color: #00ffcc !important;
        border-radius: 8px;
        border: 1px solid #00ffcc;
    }
    label, .stSelectbox label, .stNumberInput label {
        color: #ff00ff !important;
        font-weight: 500 !important;
    }
    .result-box {
        background: rgba(0, 0, 0, 0.8);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: 500;
        color: #00ffff;
        animation: glow 1.5s ease-in-out infinite alternate;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
    }
    @keyframes glow {
        from {text-shadow: 0 0 10px #00ffff;}
        to {text-shadow: 0 0 20px #ff00ff;}
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)

# Hero banner
st.image("https://user-images.githubusercontent.com/48794028/148332938-4e66d4ca-2d16-474f-8482-340aef6a48d0.png")

# Title
st.markdown("<h1 style='text-align:center; color:#00ffcc;'>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ff00ff;'>Enter property details below and get an instant price estimate.</p>", unsafe_allow_html=True)

# Dictionaries
property_condition = {"Excellent": 5, "Good": 4, "Average": 3, "Bad": 2}
quality_rating = {"Below Average": 6, "Average": 7, "Good": 8, "Very Good": 9}

# Input fields without columns
num_bedrooms = st.number_input("Bedrooms", min_value=0)
num_bathrooms = st.number_input("Bathrooms", min_value=0)
area_sqft = st.number_input("Living Area (sqft)", min_value=0)
num_floors = st.number_input("Number of Floors", min_value=0)
house_condition = st.selectbox("Condition", list(property_condition.keys()))
house_grade = st.selectbox("Grade", list(quality_rating.keys()))
postal_code = st.selectbox("Zipcode", loaded_data['zipcodes'])
house_age = st.number_input("House Age (Years)", min_value=0)

# Convert selected values
condition_score = property_condition[house_condition]
grade_score = quality_rating[house_grade]
postal_encoded = loaded_data['onehot'].transform([[postal_code]])

# Predict button
if st.button("🔍 Predict Price"):
    features_array = np.array([[num_bedrooms, num_bathrooms, area_sqft, num_floors, condition_score, grade_score, house_age]])
    combined_features = np.hstack([features_array, postal_encoded])
    scaled_features = loaded_data['scaler'].transform(combined_features)
    predicted_price = loaded_data['model'].predict(scaled_features)[0]

    st.markdown(f"<div class='result-box'>💰 Predicted Price: ₹{round(predicted_price):,}</div>", unsafe_allow_html=True)