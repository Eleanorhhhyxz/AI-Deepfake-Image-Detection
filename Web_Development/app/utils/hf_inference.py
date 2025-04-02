
import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTImageProcessor


@st.cache_resource
def _load_model(model_path: str) -> dict:
    # Load pre-trained model from local disk
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", trust_remote_code=True)
    model = ViTForImageClassification.from_pretrained(model_path)

    return {"processor": processor, "model": model}


def hf_inference(model_path: str, image: str):
    # Create both the processor and model
    vit = _load_model(model_path)

    # Process and make prediction
    inputs = vit["processor"](images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit["model"](**inputs)

    logits = outputs.logits
    return logits
