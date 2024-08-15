import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title of the app
st.title('Earthquake Prediction Using Machine Learning')

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('Earthquake_of_last_30 days.csv')

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('random_forest_model1.pkl')

data = load_data()
model = load_model()

# Display the data
st.write('### Earthquake Data of Last 30 Days', data)

# Input fields for prediction
latitude = st.number_input('Latitude:', value=0.0)
longitude = st.number_input('Longitude:', value=0.0)
depth = st.number_input('Depth (km):', value=0.0)

# Button to trigger prediction
if st.button('Predict Magnitude'):
    new_data = np.array([[latitude, longitude, depth]])
    predicted_mag = model.predict(new_data)
    st.write(f'The predicted magnitude of the earthquake using the Random Forest model is: {predicted_mag[0]}')
