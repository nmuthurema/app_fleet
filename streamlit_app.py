import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import pickle
import googlemaps
from PIL import Image
import os
import cv2
import tensorflow as tf
import gdown
from tenacity import retry, stop_after_attempt, wait_fixed

# Initialize and Configure the App
st.set_page_config(page_title="Fleet Management", layout="wide", page_icon="🚗")

# Google Maps API Key
GOOGLE_MAPS_API_KEY = "AIzaSyD9g1A2zYNNFMLfN0MReug2H5D8qjIkyNA"

# Initialize Google Maps Client
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# Define Google Drive downloader
def download_from_drive(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

# Load Models and Files
def load_model(file_path):
    try:
        if file_path.endswith('.h5'):
            return tf.keras.models.load_model(file_path)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")

def download_and_load_models():
    file_ids = {
        "pipe_model.h5": "1BUatvInxZEdFx_uRN7wHzeyVTPJis0uh",
        "tyre_model.h5": "14qoYbM_hcNhCJr6qx4T6og9348qTcl36",
        "fuel_model.pkl": "1RX_owp0qRGiJlbBFRjsTUEEVeQiRDz0v",
        "scaler.pkl": "1SuZpcvtOInDt5g9gVGlG8uSwN5j9AD4f",
        "logo.png": "1REcNnT0f0UdDar3ZzeuneLB02ufRfzFN"
    }
    models = {}
    for filename, file_id in file_ids.items():
        if not os.path.exists(filename):
            st.write(f"Downloading {filename}...")
            download_from_drive(file_id, filename)
        models[filename.split('.')[0]] = load_model(filename)
    return models

models = download_and_load_models()
pipe_model, tyre_model, fuel_model, scaler = (models['pipe_model'], models['tyre_model'], models['fuel_model'], models['scaler'])

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
def authenticate_user():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "kss@1234":  # Change to your secure credential checking
                st.session_state.authenticated = True
                st.success("Login successful!")
            else:
                st.error("Invalid username or password.")
    return st.session_state.authenticated

# Main Functionality
def main():
    if authenticate_user():
        option = st.sidebar.selectbox("Choose an option", ["Home", "Pipe Counting", "Tyre Life Prediction", "Fuel Efficiency", "Feedback"])
        
        if option == "Home":
            st.title("Welcome to Fleet Management Dashboard")
            try:
                logo_image = Image.open('logo.png')  # Attempt to load the logo image
                st.image(logo_image, width=300, caption='Logo')  # Display the logo with specified width
            except IOError:
                st.error("Error loading logo image!")  # Error message if image cannot be loaded
            
            st.markdown("Select an option from the sidebar to get started.")

        elif option == "Pipe Counting":
            perform_pipe_counting()

        elif option == "Tyre Life Prediction":
            perform_tyre_life_prediction(tyre_model)

        elif option == "Fuel Efficiency":
            calculate_fuel_efficiency()

        elif option == "Feedback":
            collect_user_feedback()

# Feature Modules
def perform_pipe_counting():
    st.title("Pipe Counting")
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        prediction = pipe_model.predict(preprocess_image(image, pipe_model))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.success(f"Number of pipes detected: {np.argmax(prediction)}")

def perform_tyre_life_prediction(tyre_model):
    st.title("Tyre Life Prediction")
    uploaded_file = st.file_uploader("Upload an image of the tyre", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_array = preprocess_image(image, tyre_model)
        prediction = tyre_model.predict(image_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        st.image(image, caption='Uploaded Tyre Image', use_column_width=True)
        st.success(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")

def calculate_fuel_efficiency():
    st.header("Fuel Requirement Estimator")
    # Similar to existing functionality for fuel efficiency

def collect_user_feedback():
    st.header("User Feedback")
    feedback = st.text_area("Share your feedback about the app:")
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback!")
        else:
            st.error("Please provide some feedback before submitting.")

if __name__ == "__main__":
    main()
