import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.row import row
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

# HEADER LAYOUT
headcol1, headcol2, headcol3, headcol4 = st.columns([0.05, 0.15, 0.05, 0.75], vertical_alignment="bottom")
with headcol2:
    st.image(os.path.join("assets", "BMKG.svg"), width=94)
with headcol4:
    st.title("Flux Machine Learning Calculator Dashboard")
st.divider()

# Input features based on model requirements
st.markdown("<h4 style='text-align: center'>Masukan variabel-variabel input Model di bawah ini untuk menghasilkan prediksi Flux</h4>", unsafe_allow_html=True)

# FEATURES LAYOUT
feature_main_container = st.container()
with feature_main_container:
    featcol1, featcol2 = st.columns(2)

# Initialize input data
input_data = {}
    
# KOLOM 1
with featcol1:
    feature1_cont = st.container()
    with feature1_cont:
        feat1img, feat1input, feat1space = st.columns([0.25, 0.75, 0.2], vertical_alignment="bottom")
    with feat1img:
        st.image(os.path.join("assets", "AirPressure.svg"), width=50)
    with feat1input:
        input_data["Tekanan Udara"] = feat1input.number_input("Tekanan Udara", value=0.0, label_visibility="visible")
    
    feature2_cont = st.container()
    with feature2_cont:
        feat2img, feat2input, feat2space = st.columns([0.25, 0.75, 0.2], vertical_alignment="bottom")
    with feat2img:
        st.image(os.path.join("assets", "Temperature.svg"), width=50)
    with feat2input:
        input_data["Suhu Avg"] = feat2input.number_input("Suhu Rata-Rata", value=0.0, label_visibility="visible")

    feature3_cont = st.container()
    with feature3_cont:
        feat3img, feat3input, feat3space = st.columns([0.25, 0.75, 0.2], vertical_alignment="bottom")
    with feat3img:
        st.image(os.path.join("assets", "Humidity.svg"), width=50)
    with feat3input:
        input_data["RH"] = feat3input.number_input("Relative Humidity", value=0.0, label_visibility="visible")
    
    feature4_cont = st.container()
    with feature4_cont:
        feat4img, feat4input, feat4space = st.columns([0.25, 0.75, 0.2], vertical_alignment="bottom")
    with feat4img:
        st.image(os.path.join("assets", "SolarRadiation.svg"), width=50)
    with feat4input:
        input_data["SR"] = feat4input.number_input("Solar Radiation", value=0.0, label_visibility="visible")

#Kolom 2 (KANAN)
with featcol2:
    feature5_cont = st.container()
    with feature5_cont:
        feat5space, feat5img, feat5input = st.columns([0.2, 0.25, 0.75], vertical_alignment="bottom")
    with feat5img:
        st.image(os.path.join("assets", "SoilTemp.svg"), width=50)
    with feat5input:
        input_data["Suhu Tanah"] = feat5input.number_input("Suhu Tanah", value=0.0, label_visibility="visible")
    
    feature6_cont = st.container()
    with feature6_cont:
        feat6space, feat6img, feat6input = st.columns([0.2, 0.25, 0.75], vertical_alignment="bottom")
    with feat6img:
        st.image(os.path.join("assets", "PH.svg"), width=50)
    with feat6input:
        input_data["PH"] = feat6input.number_input("PH", value=0.0, label_visibility="visible")
    
    feature7_cont = st.container()
    with feature7_cont:
        feat7space, feat7img, feat7input = st.columns([0.2, 0.25, 0.75], vertical_alignment="bottom")
    with feat7img:
        st.image(os.path.join("assets", "SoilMoist.svg"), width=50)
    with feat7input:
        input_data["Kelembaban Tanah"] = feat7input.number_input("Kelembaban Tanah", value=0.0, label_visibility="visible")
    
    feature8_cont = st.container()
    with feature8_cont:
        feat8space, feat8img, feat8input = st.columns([0.2, 0.25, 0.75], vertical_alignment="bottom")
    with feat8img:
        st.image(os.path.join("assets", "ElecConductive.svg"), width=50)
    with feat8input:
        input_data["EC"] = feat8input.number_input("Electrical Conductivity", value=0.0, label_visibility="visible")

ftmspace1, ftmimg, ftmtext, ftmspace2 = st.columns([0.4, 0.15, 0.15, 0.4], vertical_alignment="center")

with ftmimg:
    st.image(os.path.join("assets", "ArrowDown.svg"), width=100)

with ftmtext:
    st.html("<h3 style='text-align: left'>Feeding the model</h3>")

# Model selection
model_names = list(models.keys())
mdlspace1, mdlimg, mdlselectbox, mdlspace2 = st.columns([0.2, 0.1, 0.25, 0.2], vertical_alignment="center")

with mdlimg:
    st.image(os.path.join("assets", "MLModel.svg"), width=75)

with mdlselectbox:
    selected_model_name = st.selectbox("Pilih Model", model_names)

# Get the selected model
selected_model = models[selected_model_name]

predspace1, predimg, predtext, predspace2 = st.columns([0.4, 0.15, 0.15, 0.4], vertical_alignment="center")

with predimg:
    st.image(os.path.join("assets", "ArrowDown.svg"), width=100)

with predtext:
    st.html("<h3 style='text-align: left'>Prediction</h3>")

# # Assuming all models expect the same features (modify as needed)
# feature_names = ['Tekanan Udara', 'Suhu Average', 'RH Average', 'SR Average', 'Suhu Tanah', 'PH', 'Kelembapan Tanah', 'EC']

# # Create input fields dynamically
# input_data = {}
# for feature in feature_names:
#     input_data[feature] = st.number_input(f"{feature}", value= 0)

# Convert input dictionary to DataFrame
input_df = pd.DataFrame([input_data])

# st.write("Input Data:")
# st.write(input_df)

# Generate predictions using the selected model
if selected_model:
    try:
        prediction = selected_model.predict(input_df)
    except Exception as e:
        st.error(f"An error occurred while generating predictions: {e}")
else:
    st.error("No model is selected.")

fluxspace1, fluxtext, fluxindicator, fluxspace2 = st.columns([0.001, 0.1, 0.16, 0.006], vertical_alignment="center")

with fluxtext:
    with stylable_container(
                key="custom_container_0",
                css_styles="""
                    {
                        background-color: #639CFF;
                        border: 3px solid #639CFF;
                        border-radius: 10px;
                        align-items: center;
                        align-content: center;
                        justify-content: center;
                        justify-items: center;
                    }
                    """,
            ):
                st.html(f"<h2 style='text-align: center; font-size: 2em', 'text-color= blue'>Flux: {prediction[0]:.2f}</h2>")
                

with fluxindicator:
    if prediction > 0.14:
        with stylable_container(
                key="custom_container_1",
                css_styles="""
                    {
                        background-color: #E2E5FF;
                        border: 3px solid #639CFF;
                        border-radius: 10px;
                    }
                    """,
            ):
                indicatorspace1, indicatorimg, indicatortext, indicatorspace2 = st.columns([0.01, 0.074, 0.3, 0.01], vertical_alignment="center")
                with indicatorimg:
                    st.image(os.path.join("assets", "SmileFace.svg"), width=60)
                with indicatortext:
                    st.html(f"<h4 style='text-align: center; color: #0E1117'>Kondisi cuaca hari ini cukup cerah, sangat cocok untuk berolahraga atau sekadar berjalan-jalan.</h4>")
    else:
        with stylable_container(
                key="custom_container_2",
                css_styles="""
                    {
                        background-color: #E2E5FF;
                        border: 3px solid #639CFF;
                        border-radius: 10px;
                    }
                    """,
            ):
                indicatorspace1, indicatorimg, indicatortext, indicatorspace2 = st.columns([0.01, 0.074, 0.3, 0.01], vertical_alignment="center")
                with indicatorimg:
                    st.image(os.path.join("assets", "SadFace.svg"), width=65)
                with indicatortext:
                    st.html(f"<h4 style='text-align: center; color: #0E1117'>Kondisi cuaca hari ini kurang bagus, sebaiknya menetap di rumah dan rebahan. Stay safe!</h4>")
