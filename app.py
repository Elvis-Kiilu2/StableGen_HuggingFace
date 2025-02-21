import os
import streamlit as st
import keras
import keras_hub
import numpy as np
from PIL import Image

# Set backend
os.environ["KERAS_BACKEND"] = "jax"

# Load the model
st.title("Stable Diffusion 3 - Text-to-Image Generator")
st.write("Enter a prompt below and generate an AI-generated image!")

@st.cache_resource
def load_model():
    backbone = keras_hub.models.StableDiffusion3Backbone.from_preset(
        "stable_diffusion_3_medium", image_shape=(512, 512, 3), dtype="float16"
    )
    preprocessor = keras_hub.models.StableDiffusion3TextToImagePreprocessor.from_preset(
        "stable_diffusion_3_medium"
    )
    return keras_hub.models.StableDiffusion3TextToImage(backbone, preprocessor)

model = load_model()

# User input
prompt = st.text_input("Enter a text prompt:", "")

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            generated_image = model.generate(prompt)

            # Convert to PIL Image
            img = Image.fromarray(np.array(generated_image[0]))

            # Display image
            st.image(img, caption="Generated Image", use_column_width=True)
    else:
        st.warning("⚠️ Please enter a prompt!")
