
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the trained model, scaler, and encoders
try:
    model = joblib.load('mlp_model.joblib')
    Scaler = joblib.load('scaler.joblib')
    encoders = joblib.load('encoders.joblib')
except FileNotFoundError:
    st.error("Error: Model or transformer files not found. Please upload them.")
    st.stop()

# Define categorical and numerical columns based on the training data
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
training_cols_order = numerical_cols + categorical_cols

st.title("Adult Income Prediction App")
st.header("Enter Features for Prediction")

age = st.number_input("Age", min_value=17, max_value=75, value=30)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'NotListed', 'State-gov', 'Self-emp-inc', 'Federal-gov'])
fnlwgt = st.number_input("Fnlwgt", value=200000)
educational_num = st.number_input("Educational Number", min_value=1, max_value=16, value=9)
marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox("Occupation", ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Others', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'])
relationship = st.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
native_country = st.selectbox("Native Country", ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])

st.header("Prediction Result")
prediction_output = st.empty()

# Take user input and preprocess
user_input = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}

user_input_df = pd.DataFrame([user_input])

# Apply label encoding to user input using the fitted encoders
for col in categorical_cols:
    try:
        user_input_df[col] = encoders[col].transform(user_input_df[col])
    except ValueError as e:
        st.error(f"Error encoding '{col}': {e}. Please check the input value.")
        st.stop()

# Reorder columns to match training data order before scaling and combining
user_input_df_reordered = user_input_df[training_cols_order]

# Apply MinMaxScaler to user input using the fitted Scaler
try:
    user_input_scaled = Scaler.transform(user_input_df_reordered[numerical_cols])
except AttributeError:
    st.error("Error: MinMaxScaler 'Scaler' not found. Make sure it's defined and fitted before this step.")
    st.stop()


# Combine scaled numerical and encoded categorical features
user_input_processed = np.hstack((user_input_scaled, user_input_df_reordered[categorical_cols].values))

# Make prediction
try:
    prediction = model.predict(user_input_processed)
    prediction_output.write(f"Predicted Income: {prediction[0]}")
except AttributeError:
    st.error("Error: Machine learning model 'model' not found. Make sure it's loaded before this step.")
    st.stop()

