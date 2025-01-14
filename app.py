import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load a single model
@st.cache_resource
def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Load all models in the models/ folder
@st.cache_resource
def load_all_models(folder_path):
    models = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            model_name = os.path.splitext(filename)[0]  # Use the file name (without extension) as key
            models[model_name] = load_model(os.path.join(folder_path, filename))
    return models

# Load models
model_folder_path = "models"  # Folder containing the .pkl files
models = load_all_models(model_folder_path)

# Streamlit layout
st.title("Machine Learning Model Dashboard")
st.sidebar.header("Sidebar")

# Model selection
model_names = list(models.keys())
selected_model_name = st.sidebar.selectbox("Select a Model", model_names)

# Get the selected model
selected_model = models[selected_model_name]

# Input features based on model requirements
st.write(f"Provide input data for the model: {selected_model_name}")

# Assuming all models expect the same features (modify as needed)
feature_names = ['Tekanan Udara', 'Suhu Average', 'RH Average', 'SR Average', 'Suhu Tanah', 'PH', 'Kelembapan Tanah', 'EC']

# Create input fields dynamically
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Convert input dictionary to DataFrame
input_df = pd.DataFrame([input_data])

st.write("Input Data:")
st.write(input_df)

# Generate predictions using the selected model
if selected_model:
    try:
        st.write("Model Prediction:")
        prediction = selected_model.predict(input_df)
        st.write(prediction)
    except Exception as e:
        st.error(f"An error occurred while generating predictions: {e}")
else:
    st.error("No model is selected.")