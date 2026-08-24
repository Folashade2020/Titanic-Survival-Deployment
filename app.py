import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("titanic_gradient_boosting.joblib")
scaler = joblib.load("scaler.joblib")

st.title("🚢 Titanic Survival Predictor")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ["male", "female"])

age = st.slider("Age", 0, 100, 30)

sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)

parch = st.number_input("Parents / Children Aboard", 0, 10, 0)

fare = st.number_input("Fare", 0.0, 600.0, 32.0)

embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# One-hot encode Sex
sex_female = 1 if sex == "female" else 0
sex_male = 1 if sex == "male" else 0

# One-hot encode Embarked
embarked_c = 1 if embarked == "C" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Sex_female": [sex_female],
        "Sex_male": [sex_male],
        "Embarked_C": [embarked_c],
        "Embarked_Q": [embarked_q],
        "Embarked_S": [embarked_s]
    })

    # Scale ONLY Age and Fare
    input_data[['Age', 'Fare']] = scaler.transform(
        input_data[['Age', 'Fare']]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(
            f"🎉 Passenger would have SURVIVED!\n\nProbability: {probability:.2%}"
        )
    else:
        st.error(
            f"❌ Passenger would NOT have survived.\n\nProbability of survival: {probability:.2%}"
        )