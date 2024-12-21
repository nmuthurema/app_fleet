import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import gdown
import pickle
import os

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Fleet Management", layout="wide", page_icon="🚗")

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
    st.write("Downloading pipe model...")
    download_from_drive(PIPE_MODEL_ID, "pipe_model.h5")
if not os.path.exists("tyre_model.h5"):
    st.write("Downloading tyre model...")
    download_from_drive(TYRE_MODEL_ID, "tyre_model.h5")
if not os.path.exists("fuel_model.pkl"):
    st.write("Downloading fuel model...")
    download_from_drive(FUEL_MODEL_ID, "fuel_model.pkl")
if not os.path.exists("logo.png"):
    st.write("Downloading logo...")
    download_from_drive(LOGO_ID, "logo.png")

# Load models with error handling
try:
    pipe_model = tf.keras.models.load_model("pipe_model.h5")
except Exception as e:
    st.error(f"Failed to load pipe model: {e}")

try:
    tyre_model = tf.keras.models.load_model("tyre_model.h5")
except Exception as e:
    st.error(f"Failed to load tyre model: {e}")

try:
    with open("fuel_model.pkl", "rb") as f:
        fuel_model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load fuel model: {e}")

# Utility Functions
def preprocess_image(image, model):
    """
    Preprocess any input image to match the required shape of the given model.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        model (tf.keras.Model): The TensorFlow/Keras model to determine preprocessing.

    Returns:
        numpy.ndarray: Preprocessed image ready for model prediction.
    """
    # Get the input shape required by the model (ignoring the batch dimension)
    required_shape = model.input_shape[1:]  # (height, width, channels)

    # Resize the image to the required height and width
    image = cv2.resize(image, (required_shape[1], required_shape[0]))  # Resize to (width, height)

    # Convert to grayscale if the model expects 1 channel
    if required_shape[-1] == 1:  # Model expects grayscale input
        if len(image.shape) == 3:  # If the input is in BGR format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)  # Add a channel dimension for grayscale

    # Normalize pixel values to [0, 1]
    image = image.astype("float32") / 255.0

    # Flatten if the model expects a 1D input
    if len(required_shape) == 1:  # If input shape is 1D (e.g., (features,))
        image = image.flatten()

    # Add batch dimension
    image = np.expand_dims(image, axis=0)  # Shape becomes (1, ...)

    return image


def prepare_fuel_input(source, destination, vehicle):
    input_tensor = np.array([[len(source), len(destination), vehicle["capacity"], vehicle["fuel_efficiency"]]])
    return input_tensor

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
        try:
            logo = cv2.imread("logo.png")
            st.image(cv2.cvtColor(logo, cv2.COLOR_BGR2RGB), use_column_width=True)
        except Exception as e:
            st.error(f"Failed to load logo: {e}")
    with col2:
        st.title("Fleet Management")
        st.write("Analyze pipes, estimate fuel requirements, and predict tyre life with ease!")

    # Sidebar Navigation
    st.sidebar.title("Choose a Module")
    module = st.sidebar.radio("Modules:", ["Pipe Counting", "Fuel Requirement", "Tyre Life", "Feedback"])

    if module == "Pipe Counting" or module == "Tyre Life":
        uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

            # Select the appropriate model for prediction
            if module == "Pipe Counting":
                model = pipe_model
            elif module == "Tyre Life":
                model = tyre_model

            try:
                # Preprocess the image for the selected model
                input_tensor = preprocess_image(image, model)

                # Predict with the selected model
                prediction = model.predict(input_tensor)

                if module == "Pipe Counting":
                    num_pipes = np.argmax(prediction)
                    st.success(f"Number of pipes detected: {num_pipes}")
                elif module == "Tyre Life":
                    tyre_life = prediction[0, 0]
                    st.success(f"Estimated Tyre Life: {tyre_life:.2f} km")
            except Exception as e:
                st.error(f"Failed to make a prediction: {e}")

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
            try:
                if input_tensor.shape[1] == fuel_model.estimators_[0].n_features_in_:
                    fuel_required = fuel_model.predict(input_tensor)[0]
                    st.success(f"Estimated Fuel Required: {fuel_required:.2f} liters")
                else:
                    st.error(f"Expected {fuel_model.estimators_[0].n_features_in_} features, but got {input_tensor.shape[1]}. Check your input values.")
            except Exception as e:
                st.error(f"Failed to estimate fuel requirement: {e}")

    elif module == "Feedback":
        st.header("User Feedback")
        feedback = st.text_area("Share your feedback about the app:")
        if st.button("Submit Feedback"):
            if feedback.strip():
                st.success("Thank you for your feedback!")
            else:
                st.error("Please provide some feedback before submitting.")
