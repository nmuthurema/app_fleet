import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import gdown
import pickle
import os
from sklearn.ensemble import VotingRegressor
from torchvision import transforms

# Download models and logo from Google Drive
def download_from_drive(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

# File IDs (Replace these with actual IDs from Google Drive)
PIPE_MODEL_ID = "1BUatvInxZEdFx_uRN7wHzeyVTPJis0uh"
TYRE_MODEL_ID = "14qoYbM_hcNhCJr6qx4T6og9348qTcl36"
FUEL_MODEL_ID = "1RX_owp0qRGiJlbBFRjsTUEEVeQiRDz0v"
LOGO_ID = "1REcNnT0f0UdDar3ZzeuneLB02ufRfzFN"

# Download files (Run once)
if not os.path.exists("pipe_model.h5"):
    download_from_drive(PIPE_MODEL_ID, "pipe_model.h5")
if not os.path.exists("tyre_model.h5"):
    download_from_drive(TYRE_MODEL_ID, "tyre_model.h5")
if not os.path.exists("fuel_model.pkl"):
    download_from_drive(FUEL_MODEL_ID, "fuel_model.pkl")
if not os.path.exists("logo.png"):
    download_from_drive(LOGO_ID, "logo.png")

# Load models
pipe_model = tf.keras.models.load_model("pipe_model.h5")
tyre_model = tf.keras.models.load_model("tyre_model.h5")
with open("fuel_model.pkl", "rb") as f:
    fuel_model = pickle.load(f)

def preprocess_image(image):
    # Resize the image
    image = cv2.resize(image, (224, 224))  # Ensure size matches model input
    image = image / 255.0  # Normalize pixel values

    # Flatten the image for compatibility with the model
    image = image.reshape(-1)  # Flatten into a 1D array

    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image


def prepare_fuel_input(source, destination, vehicle):
    input_tensor = np.array([[len(source), len(destination), vehicle["capacity"], vehicle["fuel_efficiency"]]])
    return input_tensor

# Streamlit App
st.set_page_config(page_title="Vehicle & Infrastructure Tools", layout="wide", page_icon="🚗")

# Authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "kss@1234":
            st.session_state.authenticated = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password.")
else:
    # App Header
    col1, col2 = st.columns([1, 4])
    with col1:
        logo = cv2.imread("logo.png")
        st.image(cv2.cvtColor(logo, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.title("Vehicle & Infrastructure Tools")
        st.write("Analyze pipes, estimate fuel requirements, and predict tyre life with ease!")

    # Sidebar Navigation
    st.sidebar.title("Choose a Module")
    module = st.sidebar.radio("Modules:", ["Pipe Counting", "Fuel Requirement", "Tyre Life", "Feedback"])

    if module == "Pipe Counting":
        st.header("Pipe Counting")
        uploaded_file = st.file_uploader("Upload an image of the pipes:", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            input_tensor = preprocess_image(image)
            
            # Predict the number of pipes
            num_pipes = np.argmax(pipe_model.predict(input_tensor))
            st.success(f"Number of pipes detected: {num_pipes}")


    elif module == "Fuel Requirement":
    st.header("Fuel Requirement Estimator")
    source = st.text_input("Source Location:")
    destination = st.text_input("Destination Location:")

    vehicle_options = {
        "6 Wheeler (10 Tons)": {"capacity": 10, "fuel_efficiency": 6},
        "10 Wheeler (15 Tons)": {"capacity": 15, "fuel_efficiency": 5},
        "12 Wheeler (20 Tons)": {"capacity": 20, "fuel_efficiency": 3.5},
        "14 Wheeler (25 Tons)": {"capacity": 25, "fuel_efficiency": 3},
        "16 Wheeler (30 Tons)": {"capacity": 30, "fuel_efficiency": 3.2},
        "18 Wheeler (35 Tons)": {"capacity": 35, "fuel_efficiency": 2.5},
    }

    vehicle_type = st.selectbox("Type of Vehicle:", list(vehicle_options.keys()))
    selected_vehicle = vehicle_options[vehicle_type]

    if st.button("Calculate"):
        input_tensor = prepare_fuel_input(source, destination, selected_vehicle)
        
        # Validate input tensor shape
        if input_tensor.shape[1] == fuel_model.estimators_[0].n_features_in_:
            fuel_required = fuel_model.predict(input_tensor)[0]
            st.success(f"Estimated Fuel Required: {fuel_required:.2f} liters")
        else:
            st.error("Input feature count does not match the model's expected feature count.")

    # Tyre Life Module
    elif module == "Tyre Life":
        st.header("Tyre Life Prediction")
        uploaded_file = st.file_uploader("Upload an image of the tyre:", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

            input_tensor = preprocess_image(image)
            tyre_life = tyre_model.predict(input_tensor)[0, 0]
            st.success(f"Estimated Tyre Life: {tyre_life:.2f} km")

    # Feedback Section
    elif module == "Feedback":
        st.header("User Feedback")
        feedback = st.text_area("Share your feedback about the app:")
        if st.button("Submit Feedback"):
            if feedback.strip():
                st.success("Thank you for your feedback!")
            else:
                st.error("Please provide some feedback before submitting.")
