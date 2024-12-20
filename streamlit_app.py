import streamlit as st
from PIL import Image
import gdown
import os

# Download models and logo from Google Drive (replace with actual file IDs)
def download_from_drive(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

# File IDs (Replace these with actual IDs from Google Drive)
PIPE_MODEL_ID = "1ilIPLig6EVDfq-YWpsgUyEeX6ajnZkgm"
TYRE_MODEL_ID = "1REcNnT0f0UdDar3ZzeuneLB02ufRfzFN"
FUEL_MODEL_ID = "1wOdGs8Tiks-BGvAVsumtXNZz6z9Ir4qS"
LOGO_ID = "1REcNnT0f0UdDar3ZzeuneLB02ufRfzFN"
ROUTE_ID = "1PdhdlIonkZrYcjRagBTyFfqS1ZTHJUMx"

# Download files (Run once)
if not os.path.exists("pipe_model.pth"):
    download_from_drive(PIPE_MODEL_ID, "pipe_model.pth")
if not os.path.exists("tyre_model.pth"):
    download_from_drive(TYRE_MODEL_ID, "tyre_model.pth")
if not os.path.exists("fuel_model.pth"):
    download_from_drive(FUEL_MODEL_ID, "fuel_model.pth")
if not os.path.exists("logo.png"):
    download_from_drive(LOGO_ID, "logo.png")

# Load models (dummy placeholders for actual implementation)
pipe_model = "Loaded Pipe Counting Model"
tyre_model = "Loaded Tyre Life Model"
fuel_model = "Loaded Fuel Requirement Model"

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
        if username == "admin" and password == "password":  # Replace with secure auth logic
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.rerun()

        else:
            st.error("Invalid username or password.")
else:
    # App Header
    col1, col2 = st.columns([1, 4])
    with col1:
        logo = Image.open("logo.png")
        st.image(logo, use_container_width=True)
    with col2:
        st.title("Vehicle & Infrastructure Tools")
        st.write("Analyze pipes, estimate fuel requirements, and predict tyre life with ease!")

    # Sidebar Navigation
    st.sidebar.title("Choose a Module")
    module = st.sidebar.radio("Modules:", ["Pipe Counting", "Fuel Requirement", "Tyre Life", "Feedback"])

    # Pipe Counting Module
    if module == "Pipe Counting":
        st.header("Pipe Counting")
        if "pipe_uploaded_file" not in st.session_state:
            st.session_state.pipe_uploaded_file = None

        st.session_state.pipe_uploaded_file = st.file_uploader("Upload an image of the pipes:", type=["jpg", "png", "jpeg"])
        if st.session_state.pipe_uploaded_file is not None:
            image = Image.open(st.session_state.pipe_uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process image with pipe_model (dummy example)
            st.write("Processing...")
            num_pipes = 42  # Replace with actual model prediction
            st.success(f"Number of pipes detected: {num_pipes}")

        if st.button("Reset"):
            st.session_state.pipe_uploaded_file = None
            st.rerun()


    # Fuel Requirement Module
    elif module == "Fuel Requirement":
        st.header("Fuel Requirement Estimator")
        if "fuel_reset" not in st.session_state:
            st.session_state.fuel_reset = False

        source = st.text_input("Source Location:", key="source")
        destination = st.text_input("Destination Location:", key="destination")
        
        # Vehicle options with capacities and fuel efficiencies
        vehicle_options = {
            "6 Wheeler (10 Tons)": {"capacity": 10, "fuel_efficiency": 6},
            "10 Wheeler (15 Tons)": {"capacity": 15, "fuel_efficiency": 5},
            "12 Wheeler (20 Tons)": {"capacity": 20, "fuel_efficiency": 3.5},
            "14 Wheeler (25 Tons)": {"capacity": 25, "fuel_efficiency": 3},
            "16 Wheeler (30 Tons)": {"capacity": 30, "fuel_efficiency": 3.2},
            "18 Wheeler (35 Tons)": {"capacity": 35, "fuel_efficiency": 2.5},
        }
        
        vehicle_type = st.selectbox("Type of Vehicle:", list(vehicle_options.keys()), key="vehicle_type")
        selected_vehicle = vehicle_options[vehicle_type]

        # Calculate fuel requirements
        if st.button("Calculate"):
            st.write("Fetching route and fuel data...")
            distance = 191  # Example distance in km
            fuel_efficiency = selected_vehicle["fuel_efficiency"]
            fuel_required = distance / fuel_efficiency

            # Download the route image from Google Drive using ROUTE_ID
            if not os.path.exists("route.png"):
                download_from_drive(ROUTE_ID, "route.png")

            # Display results
            st.write(f"Vehicle Capacity: {selected_vehicle['capacity']} Tons")
            st.write(f"Distance: {distance} km")
            st.write(f"Fuel Efficiency: {fuel_efficiency} km/l")
            st.write(f"Fuel Required: {fuel_required:.2f} liters")

            # Display the downloaded route image
            route_image = Image.open("route.png")
            st.image(route_image, caption="Route Map", use_container_width=True)

        if st.button("Reset"):
            for key in ["source", "destination", "vehicle_type"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()



    # Tyre Life Module
    elif module == "Tyre Life":
        st.header("Tyre Life Prediction")
        if "tyre_uploaded_file" not in st.session_state:
            st.session_state.tyre_uploaded_file = None

        st.session_state.tyre_uploaded_file = st.file_uploader("Upload an image of the tyre:", type=["jpg", "png", "jpeg"])
        if st.session_state.tyre_uploaded_file is not None:
            image = Image.open(st.session_state.tyre_uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process image with tyre_model (dummy example)
            st.write("Analyzing...")
            tyre_life = "12,000 km remaining"  # Replace with actual model prediction
            st.success(f"Estimated Tyre Life: {tyre_life}")

        if st.button("Reset", key="tyre_reset"):
            st.session_state.tyre_uploaded_file = None
            st.rerun()


    # Feedback Section
    elif module == "Feedback":
        st.header("User Feedback")
        feedback = st.text_area("Share your feedback about the app:")
        if st.button("Submit Feedback"):
            if feedback.strip():
                st.success("Thank you for your feedback!")
            else:
                st.error("Please provide some feedback before submitting.")
