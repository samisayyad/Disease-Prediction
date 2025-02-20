import pickle 
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

diabetes_model_path = r"C:\Users\Asus\Desktop\disease prediction\diabetes_model.sav"
diabetes_model = pickle.load(open(diabetes_model_path,'rb'))

st.title ('Diabetes Prediction using ML')

col1,col2,col3=st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies')

with col2:
    Glucose = st.text_input('Glucose Level')

with col3:
    BloodPresure = st.text_input('BloodPresure level')

with col1:
    SkinThickness = st.text_input('SkinThickness')

with col2:
    Insulin = st.text_input('Insulin')

with col3:
    BMI = st.text_input('BMI')

with col1:
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')

with col2:
    Age = st.text_input('Age of the person')

diab_diagnosis = ''

if st.button('Diabetes test button'):
    try:
        user_input = [float(Pregnancies), float(Glucose), float(BloodPresure),float(SkinThickness), float(Insulin),
                      float(BMI), float(DiabetesPedigreeFunction), float(Age)]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = "The patient is likely to have diabetes."
        else:
            diab_diagnosis = "The patient is not likely to have diabetes."

        st.success(diab_diagnosis)

    except Exception as e:
        st.error(f"An error occurred: {e}")
