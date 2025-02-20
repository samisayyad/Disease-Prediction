import pickle
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score

# Set dark mode page title and layout
st.set_page_config(page_title="🌑 Diabetes Prediction", layout="wide", page_icon="🩺")

# Load the trained model
diabetes_model_path = r"C:\Users\Asus\Desktop\disease prediction\diabetes_model.sav"

try:
    with open(diabetes_model_path, "rb") as model_file:
        diabetes_model = pickle.load(model_file)
except Exception as e:
    st.error(f"⚠ Error loading the model: {e}")

# Dark mode CSS styling
st.markdown("""
    <style>
    body {
        background-color: #1E1E1E;
        color: #EAEAEA;
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #121212, #1A1A1A);
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 10px;
        padding: 12px;
        width: 100%;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        box-shadow: 0px 4px 8px rgba(0, 123, 255, 0.3);
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 10px;
        border: 2px solid #3A3A3A;
        background-color: #252525;
        color: #EAEAEA;
    }
    .prediction-box {
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 8px rgba(255, 255, 255, 0.1);
    }
    .header {
        text-align: center;
        color: #00E676;
        font-weight: bold;
        font-size: 30px;
    }
    .accuracy-box {
        background-color: #252525;
        color: #00E676;
        font-size: 16px;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='header'>Diabetes Prediction using ML</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Enter your details below to check your diabetes risk.</h3>", unsafe_allow_html=True)

# Load dataset for real-time accuracy calculation (Replace with actual dataset path)
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
        BloodPressure = st.text_input(" Blood Pressure", "0")

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

            st.markdown(f"<div class='accuracy-box'> Model Accuracy: <b>{real_time_accuracy*100:.2f}%</b></div>", unsafe_allow_html=True)

            # Display result with dark theme
            if diab_prediction[0] == 1:
                st.markdown("<div class='prediction-box' style='background-color: #FF5252; color: white;'> The person is DIABETIC.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='prediction-box' style='background-color: #00E676; color: #1E1E1E;'> The person is NOT DIABETIC.</div>", unsafe_allow_html=True)

        except ValueError:
            st.error("⚠ Please enter valid numerical values.")

        except Exception as e:
            st.error(f"⚠ An error occurred: {e}")

except Exception as e:
    st.error(f"⚠ Error loading dataset: {e}")