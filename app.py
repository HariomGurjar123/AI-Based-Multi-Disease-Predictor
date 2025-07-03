import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config("Multi-Disease Predictor",layout="centered")
st.markdown(
    """
    <div style="
        background-color: #1f2937;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        ">
        <h1 style="
            color: #38bdf8;
            font-size: 40px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            ">
            🧬 AI-Based Multi-Disease
        </h1>
        <h1  style="
            color: #38bdf8;
            font-size: 40px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-top: -30px;
            margin-left: 50px;
            ">Prediction System</h1>
    </div>
    """,
    unsafe_allow_html=True
)


import streamlit as st

# Custom CSS for icon buttons
st.sidebar.markdown("""
    <style>
    .icon-button {
        font-size: 24px;
        text-align: center;
        padding: 10px;
        margin: 10px 0;
        background-color: #1e293b;
        color: #38bdf8;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        width: 100%;
        display: block;
    }
    .icon-button:hover {
        background-color: #0f172a;
        color: #0ea5e9;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🔘 Choose Disease")

if "selected_disease" not in st.session_state:
    st.session_state.selected_disease = None

if st.sidebar.button("🩸 Diabetes", key="diabetes"):
    st.session_state.selected_disease = "Diabetes"
if st.sidebar.button("❤️ Heart", key="heart"):
    st.session_state.selected_disease = "Heart"
if st.sidebar.button("🎗️ Cancer", key="cancer"):
    st.session_state.selected_disease = "Cancer"
if st.sidebar.button("🦠 COVID", key="covid"):
    st.session_state.selected_disease = "COVID"
if st.sidebar.button("🧪 Liver", key="liver"):
    st.session_state.selected_disease = "Liver"

if st.session_state.selected_disease:
    st.markdown(f"""
        <div style='
            background-color: #1e293b;
            padding: 15px 25px;
            border-radius: 12px;
            margin-top: 20px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        '>
            <h3 style='
                color: #38bdf8;
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                font-size: 26px;
                margin: 0;
            '>
                ✅ Selected Disease: <span style="color: #068b90;">{st.session_state.selected_disease}</span>
            </h3>
        </div>
    """, unsafe_allow_html=True)



selected = st.session_state.get("selected_disease")

if selected:
    model_key = selected.lower().replace(" ", "_")
    model = joblib.load(f"models/{model_key}_model.pkl")
    scaler = joblib.load(f"models/{model_key}_scaler.pkl")


if selected == "Diabetes":
    st.subheader("🔹 Diabetes Prediction Form")

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
        Glucose = st.number_input("Glucose", 0, 300, step=1)
        BloodPressure = st.number_input("Blood Pressure", 0, 200, step=1)
        SkinThickness = st.number_input("Skin Thickness", 0, 100, step=1)

    with col2:
        Insulin = st.number_input("Insulin", 0, 1000, step=1)
        BMI = st.number_input("BMI", 0.0, 70.0, step=0.1)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, step=0.01)
        Age = st.number_input("Age", 1, 120, step=1)

    if st.button("🔍 Predict Diabetes"):
        input_data = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
        input_df = pd.DataFrame([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]],
                        columns=input_data)
        scaler_transform = scaler.transform(input_df)
        prediction = model.predict(scaler_transform)
        st.success(f"Prediction: {'🔴 Diabetes Detected' if prediction[0] == 1 else '🟢 No Diabetes Detected'}")

if selected == "Heart":
    st.subheader("🔹 Heart Disease Risk Prediction (SVR Based)")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", 1, 120, step=1)
        Sex = st.selectbox("Sex", ["M", "F"])
        ChestPainType = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
        RestingBP = st.number_input("Resting Blood Pressure", 0, 300, step=1)
        Cholesterol = st.number_input("Cholesterol", 0, 600, step=1)
        FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1])

    with col2:
        RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        MaxHR = st.number_input("Max Heart Rate", 60, 250, step=1)
        ExerciseAngina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
        Oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, step=0.1)
        ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # Encoding categorical values (must match training time)
    sex_val = 1 if Sex == "M" else 0
    cp_map = {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3}
    ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
    angina_map = {"N": 0, "Y": 1}
    slope_map = {"Up": 0, "Flat": 1, "Down": 2}

    cp = cp_map[ChestPainType]
    ecg = ecg_map[RestingECG]
    angina = angina_map[ExerciseAngina]
    slope = slope_map[ST_Slope]

    if st.button("🔍 Predict Heart Risk"):
        features = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]

        input_df = pd.DataFrame([[Age, sex_val, cp, RestingBP, Cholesterol,
                                  FastingBS, ecg, MaxHR, angina, Oldpeak, slope]],
                                columns=features)

        # Transform with scaler
        scaled_input = scaler.transform(input_df)

        # SVR model prediction (float value)
        prediction = model.predict(scaled_input)[0]
        percent = round(prediction * 100, 2) if prediction <= 1 else round(prediction, 2)

        # Risk level
        if percent < 25:
            stage = "🟢 Low Risk"
            color = "green"
        elif percent < 50:
            stage = "🟡 Mild Risk"
            color = "gold"
        elif percent < 75:
            stage = "🟠 Moderate Risk"
            color = "orange"
        else:
            stage = "🔴 High Risk"
            color = "red"

        st.markdown(f"""
            <div style='
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                border-left: 6px solid {color};
            '>
                <h3>💓 Risk Prediction: <span style='color:{color}'>{percent}%</span></h3>
                <h4 style="margin-left:44px">Status: {stage}</h4>
            </div>
        """, unsafe_allow_html=True)

if selected == "Cancer":
    st.subheader("🔹 Cancer Stage Prediction")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", 1, 120, step=1)
        Tumor_Size_cm = st.number_input("Tumor Size (in cm)", 0.0, 20.0, step=0.1)
        Has_Metastasis = st.selectbox("Metastasis", ["Yes", "No"])
        Family_History = st.selectbox("Family History of Cancer", ["Yes", "No"])
        Sex = st.selectbox("Sex", ["Male", "Female"])
        Cancer_Type = st.selectbox("Cancer Type", ["Carcinoma", "Leukemia", "Lymphoma", "Sarcoma"])

    with col2:
        Smoking_Habit = st.selectbox("Smoking Habit", ["Never", "Former", "Current"])
        Alcohol_Consumption = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])

    # Encode input exactly as per training features
    input_data = {
        "Age": Age,
        "Tumor_Size_cm": Tumor_Size_cm,
        "Has_Metastasis": 1 if Has_Metastasis == "Yes" else 0,
        "Family_History": 1 if Family_History == "Yes" else 0,
        "Sex_Female": 1 if Sex == "Female" else 0,
        "Sex_Male": 1 if Sex == "Male" else 0,
        "Cancer_Type_Carcinoma": 1 if Cancer_Type == "Carcinoma" else 0,
        "Cancer_Type_Leukemia": 1 if Cancer_Type == "Leukemia" else 0,
        "Cancer_Type_Lymphoma": 1 if Cancer_Type == "Lymphoma" else 0,
        "Cancer_Type_Sarcoma": 1 if Cancer_Type == "Sarcoma" else 0,
        "Smoking_Habit_Current": 1 if Smoking_Habit == "Current" else 0,
        "Smoking_Habit_Former": 1 if Smoking_Habit == "Former" else 0,
        "Smoking_Habit_Never": 1 if Smoking_Habit == "Never" else 0,
        "Alcohol_Consumption_High": 1 if Alcohol_Consumption == "High" else 0,
        "Alcohol_Consumption_Moderate": 1 if Alcohol_Consumption == "Moderate" else 0,
        "Alcohol_Consumption_None": 1 if Alcohol_Consumption == "None" else 0
    }

    input_df = pd.DataFrame([input_data])

    if st.button("🔬 Predict Cancer Stage"):
        # Load model and scaler
        model = joblib.load("models/cancer_model.pkl")
        scaler = joblib.load("models/cancer_scaler.pkl")

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        # Map output to stage
        stage_map = {
            0: ("Stage I", "#22c55e", "🟢"),
            1: ("Stage II", "#eab308", "🟡"),
            2: ("Stage III", "#f97316", "🟠"),
            3: ("Stage IV", "#ef4444", "🔴")
        }

        stage_label, color, emoji = stage_map.get(prediction, ("Unknown", "#64748b", "❓"))

        st.markdown(f"""
            <div style='
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                border-left: 6px solid {color};
            '>
                <h3> Predicted Cancer Stage: {emoji} <span style='color:{color}'>{stage_label}</span></h3>
            </div>
        """, unsafe_allow_html=True)

if selected == "COVID":
    st.subheader("🦠 COVID-19 Prediction")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", 1, 120, step=1)
        Fever = st.selectbox("Do you have Fever?", ["Yes", "No"])
        Cough = st.selectbox("Do you have Cough?", ["Yes", "No"])
        Fatigue = st.selectbox("Are you feeling Fatigue?", ["Yes", "No"])
        Loss_of_Smell = st.selectbox("Loss of Smell or Taste?", ["Yes", "No"])
        Breathing_Difficulty = st.selectbox("Breathing Difficulty?", ["Yes", "No"])

    with col2:
        Travel_History = st.selectbox("Recent Travel History?", ["Yes", "No"])
        Contact_with_COVID_Patient = st.selectbox("Contact with COVID Patient?", ["Yes", "No"])
        Comorbidity = st.selectbox("Comorbidity", ["None", "Diabetes", "Heart Disease", "Hypertension"])

    # One-hot encode manually
    input_data = {
        "Age": Age,
        "Fever": 1 if Fever == "Yes" else 0,
        "Cough": 1 if Cough == "Yes" else 0,
        "Fatigue": 1 if Fatigue == "Yes" else 0,
        "Loss_of_Smell": 1 if Loss_of_Smell == "Yes" else 0,
        "Breathing_Difficulty": 1 if Breathing_Difficulty == "Yes" else 0,
        "Travel_History": 1 if Travel_History == "Yes" else 0,
        "Contact_with_COVID_Patient": 1 if Contact_with_COVID_Patient == "Yes" else 0,
        "Comorbidity_Diabetes": 1 if Comorbidity == "Diabetes" else 0,
        "Comorbidity_Heart Disease": 1 if Comorbidity == "Heart Disease" else 0,
        "Comorbidity_Hypertension": 1 if Comorbidity == "Hypertension" else 0,
        "Comorbidity_None": 1 if Comorbidity == "None" else 0
    }

    input_df = pd.DataFrame([input_data])

    if st.button("🔬 Predict COVID-19"):
        model = joblib.load("models/covid_model.pkl")
        scaler = joblib.load("models/covid_scaler.pkl")

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else None

        color = "#ef4444" if prediction == 1 else "#22c55e"
        status = "🦠 COVID POSITIVE" if prediction == 1 else "✅ COVID NEGATIVE"
        emoji = "🔴" if prediction == 1 else "🟢"

        st.markdown(f"""
            <div style='
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                border-left: 6px solid {color};
            '>
                <h3>{emoji} Prediction: <span style='color:{color}'>{status}</span></h3>
            </div>
        """, unsafe_allow_html=True)

if selected == "Liver":
    st.subheader("🧪 Liver Disease Prediction")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", 1, 120, step=1)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Total_Bilirubin = st.number_input("Total Bilirubin", 0.0, 75.0, step=0.1)
        Direct_Bilirubin = st.number_input("Direct Bilirubin", 0.0, 25.0, step=0.1)
        Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase", 0, 2000, step=1)

    with col2:
        Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase", 0, 2000, step=1)
        Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase", 0, 2000, step=1)
        Total_Protiens = st.number_input("Total Proteins", 0.0, 10.0, step=0.1)
        Albumin = st.number_input("Albumin", 0.0, 5.0, step=0.1)
        Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio", 0.0, 2.5, step=0.01)

    if st.button("🔍 Predict Liver Disease"):
        # Map Gender to numeric like model training (Male = 1, Female = 0)
        gender_numeric = 1 if Gender == "Male" else 0

        # Prepare input in same order as model training
        input_data = {
            "Age": Age,
            "Gender": gender_numeric,
            "Total_Bilirubin": Total_Bilirubin,
            "Direct_Bilirubin": Direct_Bilirubin,
            "Alkaline_Phosphotase": Alkaline_Phosphotase,
            "Alamine_Aminotransferase": Alamine_Aminotransferase,
            "Aspartate_Aminotransferase": Aspartate_Aminotransferase,
            "Total_Protiens": Total_Protiens,
            "Albumin": Albumin,
            "Albumin_and_Globulin_Ratio": Albumin_and_Globulin_Ratio
        }

        input_df = pd.DataFrame([input_data])

        # Load model and scaler
        model = joblib.load("models/liver_model.pkl")
        scaler = joblib.load("models/liver_scaler.pkl")

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else None

        color = "#ef4444" if prediction == 1 else "#22c55e"
        status = "🧪 Liver Disease Detected" if prediction == 1 else "✅ No Liver Disease"
        emoji = "🔴" if prediction == 1 else "🟢"

        st.markdown(f"""
            <div style='
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                border-left: 6px solid {color};
            '>
                <h3>{emoji} Prediction: <span style='color:{color}'>{status}</span></h3>
                {"<p>Probability of Liver Disease: {:.2f}%</p>".format(probability*100) if probability is not None else ""}
            </div>
        """, unsafe_allow_html=True)
