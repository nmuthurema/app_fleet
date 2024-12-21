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

# Set up your page configuration and Google Maps API
st.set_page_config(page_title="Fleet Management", layout="wide", page_icon="🚗")
GOOGLE_MAPS_API_KEY = "AIzaSyD9g1A2zYNNFMLfN0MReug2H5D8qjIkyNA"
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# Define Google Drive downloader function
def download_from_drive(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

# Load Models and Files
def load_model(file_path):
    if file_path.endswith('.h5'):
        return tf.keras.models.load_model(file_path)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

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
            download_from_drive(file_id, filename)
        models[filename.split('.')[0]] = load_model(filename)
    return models

models = download_and_load_models()
pipe_model, tyre_model, fuel_model, scaler = (models['pipe_model'], models['tyre_model'], models['fuel_model'], models['scaler'])

# Utility Functions
def preprocess_image(image, model):
    # Check if the image is a PIL Image and convert it to a NumPy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)

    required_shape = model.input_shape[1:]

    # Resize the image to match the input shape required by the model
    image = cv2.resize(image, (required_shape[1], required_shape[0]))

    # Handle grayscale conversion if the model expects one channel
    if required_shape[-1] == 1:
        if len(image.shape) == 3:  # Color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale

    image = image.astype("float32") / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

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

def authenticate_user():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    with st.sidebar:
        if not st.session_state['authenticated']:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if username == "admin" and password == "kss@1234":  # Validate credentials
                    st.session_state['authenticated'] = True
                    st.success("Login successful!")
                else:
                    st.error("Incorrect username or password.")

    if not st.session_state['authenticated']:
        st.sidebar.warning("Please log in to continue.")

    return st.session_state['authenticated']

def get_tyres_lifetime_and_maintenance(predicted_class):
    lifetime_mapping = {
        0: {
            "lifetime": "5 years",  # class_1 (New Tyres)
            "maintenance": "Inspect every 6 months for wear and tear. Ensure proper inflation levels and alignment."
        },
        1: {
            "lifetime": "3 years",  # class_2 (Cracked Tyres)
            "maintenance": "Inspect monthly for cracks and damage. Consider rotating tyres to even out wear. Replace if cracks deepen."
        },
        2: {
            "lifetime": "1 year",  # class_3 (Severely Cracked Tyres)
            "maintenance": "Replace immediately to avoid safety risks. Avoid driving until the replacement is done."
        }
    }
    return lifetime_mapping.get(predicted_class, {"lifetime": "Unknown Class", "maintenance": "No suggestions available"})

def perform_tyre_life_prediction(tyre_model):
    st.title("Tyre Life Prediction")
    uploaded_file = st.file_uploader("Upload an image of the tyre", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_array = preprocess_image(image, tyre_model)
        prediction = tyre_model.predict(image_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Display the image and the results
        st.image(image, caption='Uploaded Tyre Image', use_column_width=True)
        result_text = f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%"
        st.success(result_text)

        # Fetch and display the lifetime and maintenance suggestions
        tyre_info = get_tyres_lifetime_and_maintenance(predicted_class)
        st.subheader("Estimated Tyre Lifetime and Maintenance")
        st.write(f"**Estimated Lifetime:** {tyre_info['lifetime']}")
        st.write(f"**Maintenance Suggestion:** {tyre_info['maintenance']}")

def perform_pipe_counting(pipe_model):
    st.title("Pipe Counting")
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        try:
            # Open the image file and convert it to a format compatible with OpenCV
            image = Image.open(uploaded_file).convert('RGB')
            image_array = np.array(image)  # Convert image to numpy array

            if pipe_model:
                # Preprocess the image
                processed_image = preprocess_image(image_array, pipe_model)

                # Predict the number of pipes in the image using the model
                prediction = pipe_model.predict(processed_image)

                # Assuming the model returns a direct count, not classifications
                predicted_count = int(prediction.flatten()[0])

                # Display the uploaded image and the prediction result
                st.image(image, caption='Uploaded Image', use_column_width=True)
                st.success(f"Number of pipes detected: {predicted_count}")
            else:
                st.error("Pipe model is not loaded properly.")

        except Exception as e:
            st.error(f"Failed to process the image or prediction: {e}")

def collect_user_feedback():
    st.header("User Feedback")
    feedback = st.text_area("Share your feedback about the app:")
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback!")
        else:
            st.error("Please provide some feedback before submitting.")
            
def main():
    if authenticate_user():
        option = st.sidebar.selectbox("Choose an option", ["Home", "Pipe Counting", "Tyre Life Prediction", "Fuel Efficiency", "Feedback"])
        
        if option == "Home":
            st.title("Welcome to Fleet Management Dashboard")
            try:
                logo_image = Image.open('logo.png')
                st.image(logo_image, width=300, caption='Fleet Management System')
            except IOError:
                st.error("Error loading logo image!")
            st.markdown("Select an option from the sidebar to get started.")

        elif option == "Pipe Counting":
            perform_pipe_counting(pipe_model)

        elif option == "Tyre Life Prediction":
            perform_tyre_life_prediction(tyre_model)

        elif option == "Fuel Efficiency":
            
            st.header("Fuel Requirement Estimator")

            # User Inputs
            source = st.text_input("Source Location:")
            destination = st.text_input("Destination Location:")
            actual_weight = st.number_input("Enter weight of goods (in tons):", min_value=7.5, value=20.0)
            number_of_wheels = st.selectbox("Number of Wheels:", [6, 10, 12, 14, 16, 18])
            
            # Mileage table as provided
            mileage_table = pd.DataFrame({
                "No of wheels": [6, 8, 10, 12, 14, 16, 18],
                "Tonnage": [7.5, 13.5, 20, 25, 30, 35, 40],
                "One Side Upper": [7.5, 6.5, 5.5, 5, 4.5, 4.5, 4],
                "One Side Lower": [6.5, 5.5, 4, 3.5, 3.2, 3, 2],
                "Two Sides Upper": [6.5, 5.7, 5, 4.7, 4.2, 4.2, 3.7],
                "Two Sides Lower": [5.5, 4.7, 3.5, 3.2, 2.9, 2.7, 1.7]
            })
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
                        loading_condition = st.radio(
                            "Is the vehicle loaded on one side or both sides?",
                            ("One Side Loaded", "Both Sides Loaded")
                        )
            
                        # Retrieve mileage limits based on user input
                        mileage_info = mileage_table[mileage_table["No of wheels"] == number_of_wheels].iloc[0]
                        if loading_condition == "One Side Loaded":
                            min_mileage, max_mileage = mileage_info["One Side Lower"], mileage_info["One Side Upper"]
                        else:
                            min_mileage, max_mileage = mileage_info["Two Sides Lower"], mileage_info["Two Sides Upper"]

                        # Prepare Data for Scaler
                        try:
                            input_data = [distance, actual_weight, number_of_wheels, capacity, mileage]
                            input_data_scaled = scaler.transform([input_data])

                            # Predict Fuel Efficiency
                            predicted_efficiency = fuel_model.predict(input_data_scaled)[0]
                            predicted_efficiency = min(max(predicted_efficiency, min_mileage), max_mileage)

                            # Calculate Fuel Requirement
                            predicted_fuel = distance / predicted_efficiency
                            st.success(f"Predicted Fuel Requirement: {predicted_fuel:.2f} liters")
                            st.info(f"Predicted Fuel Efficiency: {predicted_efficiency:.2f} km/l")
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")


        elif option == "Feedback":
            collect_user_feedback()
    else:
        st.sidebar.warning("Please login to continue.")

if __name__ == "__main__":
    main()
