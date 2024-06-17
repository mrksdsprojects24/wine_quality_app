import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
#data = pd.read_csv("/content/gdrive/MyDrive/winequality-white.csv", sep=";")
data1 = pd.read_csv("winequality-white.csv", sep=";")
data2 = pd.read_csv("winequality-red.csv", sep=";")
data = pd.concat([data1, data2], axis=0)

X = data.drop("quality", axis=1)
y = data["quality"]  # Target variable

# Define Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Set number of trees (n_estimators)

# Train the model
model.fit(X, y)

# Define a function to predict wine quality
def predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol):
    new_data = pd.DataFrame({
        "fixed acidity": [fixed_acidity],
        "volatile acidity": [volatile_acidity],
        "citric acid": [citric_acid],
        "residual sugar": [residual_sugar],
        "chlorides": [chlorides],
        "free sulfur dioxide": [free_sulfur_dioxide],
        "total sulfur dioxide": [total_sulfur_dioxide],
        "density": [density],
        "pH": [ph],
        "sulphates": [sulphates],
        "alcohol": [alcohol]
    })
    prediction = model.predict(new_data)[0]
    return prediction

# Build the Streamlit app
import streamlit as st

st.title("Krishna's Wine Quality Prediction App")
st.write("Please input as many properties of your wine as possible (as you know). Each property is already given a default value equal to the median value in the dataset used to train the machine learning algorithm. When you ready, click the Predict-Wine-Quality button to find the quality of your wine.")

fixed_acidity = st.number_input("Fixed Acidity", min_value=3.8, max_value=16.0, value=7.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.08, max_value=1.56, value=0.29)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.66, value=0.31)
residual_sugar = st.number_input("Residual Sugar", min_value=0.6, max_value=66.0, value=3.0)
chlorides = st.number_input("Chlorides", min_value=0.009, max_value=0.61, value=0.047)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=1.0, max_value=290.0, value=29.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=6.0, max_value=440.0, value=118.0)
density = st.number_input("Density", min_value=0.980, max_value=1.038, value=0.995)
ph = st.number_input("pH", min_value=2.7, max_value=4.0, value=3.21)
sulphates = st.number_input("Sulphates", min_value=0.22, max_value=2.0, value=0.51)
alcohol = st.number_input("Alcohol", min_value=8.0, max_value=14.9, value=10.3)

if st.button("Predict Wine Quality"):
    prediction = predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol)
    st.write("As per my machine learning algorithm trained on a dataset xxx, the quality of your wine (on a scale of 1-10) is:")
    st.write(f"{prediction}")