
import streamlit as st
import numpy as np
import timm
import torchvision.transforms as transforms
import torch
from PIL import Image
from tensorflow.keras.models import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_data
def image_generator(
    image: Image, image_width: int, image_height: int, normalize: bool, model_type: str
) -> np.ndarray | torch.Tensor:
    if model_type == "HDF5":
        # Perform image preprocessing similar to training set
        image = image.resize((image_height, image_width))
        img_array = np.array(image)
        if normalize:
            img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    elif model_type == "Hugging Face":
        transform_pipeline = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.CenterCrop(image_height)
        ])
        img_array = transform_pipeline(image).unsqueeze(0)

    else:
        transform_pipeline = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_array = transform_pipeline(image).unsqueeze(0)

    return img_array


@st.cache_data
def custom_generator(
    image: Image, image_width: int, image_height: int, normalize: bool
) -> tuple:
    # Convert image to grey scale
    image_grey = image.convert("L")

    # Get the frequency domain representation
    img_freq = np.array(image_grey.resize((image_height, image_width)))
    dft = np.fft.fft2(img_freq)
    magnitude_spectrum = np.log(1 + np.abs(dft))

    # Get the spacial domain representation
    img_space = np.array(image.resize((image_height, image_width)))
    if normalize:
        img_space = img_space / 255.0

    return magnitude_spectrum[..., np.newaxis], img_space


@st.cache_resource
def _model_load(model_path: str, model_type: str, base_model: str | None):
    # Tensorflow model
    if model_type == "HDF5" or model_type == "Custom":
        # Load the model and set the model to not trainable
        model = load_model(model_path)
        model.trainable = False

    # Pytorch model
    else:
        # Load the model and set the model to evaluation mode
        model = timm.create_model(base_model, pretrained=False, num_classes=1)
        model_checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(model_checkpoint["model_state_dict"])

    return model


def model_prediction(model_path: str, img_arr: np.ndarray | torch.Tensor | list, model_type: str, base_model: str | None):
    # Load the pre-trained models
    model = _model_load(model_path, model_type, base_model)

    # Tensorflow model
    if model_type == "HDF5" or model_type == "Custom":
        # Run inference on the image
        predictions = model.predict(img_arr)

    # Pytorch model
    else:
        model.eval().to(DEVICE)

        # Send the img_array to device
        img_tensor = img_arr.to(DEVICE)

        # Run the inference and get prediction
        with torch.no_grad():
            predictions = model(img_tensor)

        predictions = torch.sigmoid(predictions)

    return predictions

