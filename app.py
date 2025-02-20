import pickle
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score

# Set page title and layout
st.set_page_config(page_title="💉 Diabetes Prediction", layout="wide", page_icon="🌟")

# Load the trained model
diabetes_model_path = r"C:\Users\Asus\Desktop\disease prediction\diabetes_model.sav"

try:
    with open(diabetes_model_path, "rb") as model_file:
        diabetes_model = pickle.load(model_file)
except Exception as e:
    st.error(f"⚠ Error loading the model: {e}")

# Modern CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    body {
        background: black);
        color: #F8F9FA;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: transparent;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        color: white;
        font-size: 16px;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 14px 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 6px 15px rgba(0, 201, 255, 0.6);
    }
    .stTextInput>div>div>input {
        background-color: #303344;
        color: #F8F9FA;
        border: 2px solid #44475A;
        border-radius: 10px;
        padding: 10px;
        font-size: 15px;
    }
    .stTextInput>div>div>input::placeholder {
        color: #BFBFBF;
    }
    .header {
        text-align: center;
        color: #92FE9D;
        font-weight: bold;
        font-size: 36px;
        margin-bottom: 10px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
    }
    .subheader {
        text-align: center;
        color: #A1CAF1;
        font-size: 20px;
        margin-bottom: 30px;
    }
    .accuracy-box {
        background: #2B2D42;
        color: #00FFC6;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 3px 8px rgba(0, 0, 0, 0.3);
    }
    .prediction-box {
        font-size: 20px;
        font-weight: bold;
        padding: 16px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);
    }
    .diabetic {
        background-color: #FF5E57;
        color: white;
    }
    .not-diabetic {
        background-color: #3ECF8E;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Subheader
st.markdown("<h1 class='header'>💉 Diabetes Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subheader'>Enter your details below to assess your diabetes risk</h3>", unsafe_allow_html=True)

# Load dataset for real-time accuracy calculation
dataset_path = r"C:\Users\Asus\Desktop\disease prediction\diabetes.csv"

try:
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=["Outcome"])  # Features
    y = data["Outcome"]  # Target (0: Non-Diabetic, 1: Diabetic)

    # Calculate model accuracy
    y_pred = diabetes_model.predict(X)
    real_time_accuracy = accuracy_score(y, y_pred)

    # Input form using Streamlit columns
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies", "0")

    with col2:
        Glucose = st.text_input("Glucose Level", "0")

    with col3:
        BloodPressure = st.text_input("Blood Pressure", "0")

    with col1:
        SkinThickness = st.text_input("Skin Thickness", "0")

    with col2:
        Insulin = st.text_input("Insulin Level", "0")

    with col3:
        BMI = st.text_input("BMI Value", "0.0")

    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", "0.0")

    with col2:
        Age = st.text_input("Age", "0")

    # Prediction Button
    if st.button("Predict Diabetes"):
        try:
            # Convert user inputs safely
            user_input = np.array([
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]).reshape(1, -1)

            # Predict diabetes
            diab_prediction = diabetes_model.predict(user_input)

            st.markdown(f"<div class='accuracy-box'>Model Accuracy: <b>{real_time_accuracy*100:.2f}%</b></div>", unsafe_allow_html=True)

            # Display result with modern styling
            if diab_prediction[0] == 1:
                st.markdown("<div class='prediction-box diabetic'> The person is DIABETIC.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='prediction-box not-diabetic'>The person is NOT DIABETIC.</div>", unsafe_allow_html=True)

        except ValueError:
            st.error("⚠ Please enter valid numerical values.")

        except Exception as e:
            st.error(f"⚠ An error occurred: {e}")

except Exception as e:
    st.error(f"⚠ Error loading dataset: {e}")
