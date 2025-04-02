
import streamlit as st
import numpy as np
import yaml
import torch
from pathlib import Path
from PIL import Image
from utils import model_inference, hf_inference

# Read in configuration file
CONFIG_FILE = Path(__file__).joinpath("..", "..", "config.yaml").resolve()
CONFIG = yaml.safe_load(open(CONFIG_FILE, mode="r"))

# Get GIF URL
FAKE_URL = "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExcnNzeGFzcTd2bmFzYjc2NTY4NzQwMmd2MWJ4YXloaWFpbDUydzN3ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/OqAeQrGmU7lS6tENnQ/giphy.gif"
REAL_URL = "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2E5M2Q3azh1Y2dhYXNoeTEwMnhldXA3aHQ4Mmt4MXhya2poOHE3OCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/qZMlsJ9AW2J4Isc8nC/giphy.gif"

# Set page configuration for entry point
st.set_page_config(page_title="AI vs. Human Image Detection", page_icon=":robot_face:")
st.sidebar.header("Image Classification")
st.sidebar.markdown("AI vs. Human Image Detection")


# ---------------------------------------------------------------------------------------------------------------------
# Application Main Page
# ---------------------------------------------------------------------------------------------------------------------

# Title of this main page
st.write("## AI vs. Human Image Detection :robot_face:")

# Add markdown to describe the page and how to use
st.markdown(
    """
    In today’s rapidly evolving digital landscape, AI-generated content is becoming more sophisticated and widespread.
    From deep-fakes to generative art, distinguishing between what’s real and what’s AI is a growing challenge with
    profound implications for media, security, and creativity.
    
    Let's evaluate the ability of some fine-tuned Deep Learning models to distinguish between real and AI images.
    
    --- 
    """
)

# Add markdown on instructions on how to upload images and pick a model of choice
st.markdown(
    """
    To start, pick a model from the dropdown menu. Then, upload an image for classification.
    """
)


# ---------------------------------------------------------------------------------------------------------------------
# Section for dropdown box and image upload
# ---------------------------------------------------------------------------------------------------------------------

# Get model choices from configuration file
MODEL_CHOICES = CONFIG["Image_Classification_Models"]

st.write("#### Model Selection")
option = st.selectbox(
    label="Please select a model.",
    options=[key for key in MODEL_CHOICES.keys()],
    placeholder="Select a model..."
)

# Create a section for user to upload images
st.write("#### Image Upload")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Show the image if there is an upload
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

# End the section
st.markdown("---")


# ---------------------------------------------------------------------------------------------------------------------
# Load model, standardize image, and make prediction
# ---------------------------------------------------------------------------------------------------------------------

if uploaded_file is not None:
    model_type = MODEL_CHOICES[option]["Model_Type"]
    base_model = MODEL_CHOICES[option].get("Base_Model", None)

    # Get image dimension and preprocessing configs
    image_width, image_height = MODEL_CHOICES[option]["Image_Width"], MODEL_CHOICES[option]["Image_Height"]
    normalize = MODEL_CHOICES[option]["Normalize"]
    model_file = MODEL_CHOICES[option]["File_Name"]

    # Based on the model selection, load the model into memory
    model_path = Path(__file__).joinpath("..", "..", "models", "image_classification", model_file).resolve().__str__()

    # Make prediction of if the image is AI generated or human generated
    if model_type == "HDF5" or model_type == "Torch":
        # Preprocess image
        preprocessed_image = model_inference.image_generator(image, image_width, image_height, normalize, model_type)

        # Make prediction
        prediction = model_inference.model_prediction(model_path, preprocessed_image, model_type, base_model)
        classification = "AI" if prediction > 0.5 else "Human"

    elif model_type == "Custom":
        # Preprocess image
        img_freq, img_spacial = model_inference.custom_generator(image, image_width, image_height, normalize)
        img_freq = np.expand_dims(img_freq, axis=0)
        img_spacial = np.expand_dims(img_spacial, axis=0)

        # Make prediction
        prediction = model_inference.model_prediction(model_path, [img_freq, img_spacial], model_type, base_model)
        classification = "AI" if prediction > 0.5 else "Human"

    else:
        # Process image
        preprocessed_image = model_inference.image_generator(image, image_width, image_height, normalize, model_type)

        # Make prediction
        logits = hf_inference.hf_inference(model_path, preprocessed_image)
        prediction = logits.argmax(dim=1)
        classification = "AI" if prediction == 1 else "Human"

    # Based on the classification, give users a response
    if classification == "AI":
        st.markdown("### The image you uploaded is most likely AI generated!")
        st.image(FAKE_URL, use_container_width=True)

    else:
        st.markdown("### The image you uploaded is most likely human generated. You are good!")
        st.image(REAL_URL, use_container_width=True)
