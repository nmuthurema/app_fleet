import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import pickle
import googlemaps
from PIL import Image
from packaging import version
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
from tenacity import retry, stop_after_attempt, wait_fixed
import cv2
import tensorflow as tf
import gdown

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Fleet Management", layout="wide", page_icon="🚗")

# Google Maps API Key
GOOGLE_MAPS_API_KEY = "AIzaSyD9g1A2zYNNFMLfN0MReug2H5D8qjIkyNA"
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# Download models and logo from Google Drive
def download_from_drive(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

# File IDs (Replace these with actual IDs from Google Drive)
PIPE_MODEL_ID = "1BUatvInxZEdFx_uRN7wHzeyVTPJis0uh"
TYRE_MODEL_ID = "14qoYbM_hcNhCJr6qx4T6og9348qTcl36"
FUEL_MODEL_ID = "1RX_owp0qRGiJlbBFRjsTUEEVeQiRDz0v"
SCALER_ID = "1SuZpcvtOInDt5g9gVGlG8uSwN5j9AD4f"
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
if not os.path.exists("scaler.pkl"):
    st.write("Downloading standard scaler...")
    download_from_drive(SCALER_ID, "scaler.pkl")
if not os.path.exists("logo.png"):
    st.write("Downloading logo...")
    download_from_drive(LOGO_ID, "logo.png")

# Load models with error handling
try:
    # Validate file existence and size
    if not os.path.exists("pipe_model.h5") or os.path.getsize("pipe_model.h5") == 0:
        st.write("Downloading pipe model...")
        download_from_drive(PIPE_MODEL_ID, "pipe_model.h5")
        if not os.path.exists("pipe_model.h5") or os.path.getsize("pipe_model.h5") == 0:
            raise ValueError("Pipe model file is empty or corrupted.")

    # Load the model
    pipe_model = tf.keras.models.load_model("pipe_model.h5")
except tf.errors.OpError as e:
    st.error(f"TensorFlow error while loading pipe model: {e}")
except ValueError as e:
    st.error(f"Validation error: {e}")
except Exception as e:
    st.error(f"Unexpected error while loading pipe model: {e}")

try:
    # Validate file existence and size
    if not os.path.exists("tyre_model.h5") or os.path.getsize("tyre_model.h5") == 0:
        st.write("Downloading tyre model...")
        download_from_drive(TYRE_MODEL_ID, "tyre_model.h5")
        if not os.path.exists("tyre_model.h5") or os.path.getsize("tyre_model.h5") == 0:
            raise ValueError("Tyre model file is empty or corrupted.")

    # Load the model
    tyre_model = tf.keras.models.load_model("tyre_model.h5")
except tf.errors.OpError as e:
    st.error(f"TensorFlow error while loading tyre model: {e}")
except ValueError as e:
    st.error(f"Validation error: {e}")
except Exception as e:
    st.error(f"Unexpected error while loading tyre model: {e}")

try:
    with open("fuel_model.pkl", "rb") as f:
        fuel_model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
except Exception as e:
    st.error(f"Failed to load fuel model or scaler: {e}")

# Utility Functions
def preprocess_image(image, model):
    required_shape = model.input_shape[1:]
    image = cv2.resize(image, (required_shape[1], required_shape[0]))
    if required_shape[-1] == 1:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
    image = image.astype("float32") / 255.0
    if len(required_shape) == 1:
        image = image.flatten()
    image = np.expand_dims(image, axis=0)
    return image

def get_coordinates(location):
    try:
        geocode_result = gmaps.geocode(location)
        if geocode_result:
            lat = geocode_result[0]["geometry"]["location"]["lat"]
            lon = geocode_result[0]["geometry"]["location"]["lng"]
            return lat, lon
        else:
            st.error(f"Could not find coordinates for {location}")
            return None, None
    except Exception as e:
        st.error(f"Error fetching coordinates: {e}")
        return None, None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def calculate_distance(source_coords, destination_coords):
    try:
        directions_result = gmaps.directions(
            origin=source_coords,
            destination=destination_coords,
            mode="driving"
        )
        if directions_result:
            distance = directions_result[0]['legs'][0]['distance']['value'] / 1000
            route_coords = [
                (step["start_location"]["lat"], step["start_location"]["lng"])
                for step in directions_result[0]["legs"][0]["steps"]
            ]
            route_coords.append((destination_coords[0], destination_coords[1]))
            return distance, route_coords
        else:
            st.error("Could not find a route between the locations.")
            return None, None
    except Exception as e:
        st.error(f"Error calculating distance: {e}")
        return None, None

def display_map(source_coords, destination_coords, route_coords=None):
    map_center = (
        (source_coords[0] + destination_coords[0]) / 2,
        (source_coords[1] + destination_coords[1]) / 2
    )
    m = folium.Map(location=map_center, zoom_start=12)
    folium.Marker(location=source_coords, popup="Source", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(location=destination_coords, popup="Destination", icon=folium.Icon(color="red")).add_to(m)
    if route_coords:
        folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=1).add_to(m)
    return m

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
    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            # Load the image using cv2
            logo = cv2.imread("logo.png")
    
            # Convert BGR (OpenCV format) to RGB
            logo_rgb = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
    
            # Convert to PIL Image for Streamlit compatibility
            logo_image = Image.fromarray(logo_rgb)
    
            # Display the logo with version compatibility
            st.image(logo_image, use_column_width=True)

        except Exception as e:
            st.error(f"Failed to load logo: {e}")

    with col2:
        st.title("Fleet Management")
        st.write("Analyze pipes, estimate fuel requirements, and predict tyre life with ease!")

    st.sidebar.title("Choose a Module")
    module = st.sidebar.radio("Modules:", ["Pipe Counting", "Fuel Requirement", "Tyre Life", "Feedback"])

    if module in ["Pipe Counting", "Tyre Life"]:
        # File upload
        uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            try:
                # Read the uploaded image as bytes
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                
                # Decode the image using OpenCV
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
            
            except cv2.error as e:
                st.error(f"OpenCV error while processing the image: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                # Select appropriate model
                model = pipe_model if module == "Pipe Counting" else tyre_model

                # Preprocess the image
                input_tensor = preprocess_image(image, model)

                # Make predictions
                prediction = model.predict(input_tensor)

                # Display results based on module type
                if module == "Pipe Counting":
                    st.success(f"Number of pipes detected: {np.argmax(prediction)}")
                elif module == "Tyre Life":
                    st.success(f"Estimated Tyre Life: {prediction[0, 0]:.2f} km")

            except cv2.error as e:
                st.error(f"Failed to process the image: OpenCV Error - {e}")
            except ValueError as e:
                st.error(f"Image preprocessing failed: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


    elif module == "Fuel Requirement":
        st.header("Fuel Requirement Estimator")

        # User Inputs
        source = st.text_input("Source Location:")
        destination = st.text_input("Destination Location:")
        actual_weight = st.number_input("Enter weight of goods (in tons):", min_value=7.5, value=20.0)
        number_of_wheels = st.selectbox("Number of Wheels:", [6, 10, 12, 14, 16, 18])

        # Vehicle Options: number_of_wheels determines capacity and mileage
        vehicle_options = {
            6: {"capacity": 7.5, "one_side_mileage": 7, "both_side_mileage": 6},
            10: {"capacity": 13.5, "one_side_mileage": 5.5, "both_side_mileage": 4.5},
            12: {"capacity": 20, "one_side_mileage": 4.5, "both_side_mileage": 3.2},
            14: {"capacity": 25, "one_side_mileage": 4.2, "both_side_mileage": 3},
            16: {"capacity": 30, "one_side_mileage": 3.7, "both_side_mileage": 2.5},
            18: {"capacity": 35, "one_side_mileage": 3.5, "both_side_mileage": 2.3},
        }

        selected_vehicle = vehicle_options[number_of_wheels]
        capacity = selected_vehicle["capacity"]  # Extract capacity
        loading_condition = st.radio(
            "Is the vehicle loaded on one side or both sides?",
            ("One Side Loaded", "Both Sides Loaded")
        )

        # Calculate Mileage Based on Loading Condition
        mileage = (
            selected_vehicle["one_side_mileage"]
            if loading_condition == "One Side Loaded"
            else selected_vehicle["both_side_mileage"]
        )

        if source and destination:
            source_coords = get_coordinates(source)
            destination_coords = get_coordinates(destination)

            if source_coords and destination_coords:
                distance, route_coords = calculate_distance(source_coords, destination_coords)

                if distance:
                    st.write(f"Calculated Distance: {distance:.2f} km")
                    st.write("Route Map:")
                    route_map = display_map(source_coords, destination_coords, route_coords)
                    folium_static(route_map)

                    # Prepare Data for Scaler
                    try:
                        input_data = [distance, actual_weight, number_of_wheels, capacity, mileage]
                        input_data_scaled = scaler.transform([input_data])

                        # Predict Fuel Efficiency
                        predicted_efficiency = fuel_model.predict(input_data_scaled)[0]

                        # Constrain efficiency to realistic bounds
                        max_efficiency = 6 if selected_vehicle["capacity"] <= 15 else 3.5
                        predicted_efficiency = min(max_efficiency, max(1, predicted_efficiency))

                        # Calculate Fuel Requirement
                        predicted_fuel = distance / predicted_efficiency
                        st.success(f"Predicted Fuel Requirement: {predicted_fuel:.2f} liters")
                        st.info(f"Predicted Fuel Efficiency: {predicted_efficiency:.2f} km/l")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")


    elif module == "Feedback":
        st.header("User Feedback")
        feedback = st.text_area("Share your feedback about the app:")
        if st.button("Submit Feedback"):
            if feedback.strip():
                st.success("Thank you for your feedback!")
            else:
                st.error("Please provide some feedback before submitting.")
