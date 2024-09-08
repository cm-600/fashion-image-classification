import os
import streamlit as st
from huggingface_hub import login
from transformers import pipeline
from PIL import Image

hf_token = st.secrets["HUGGINGFACE_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

login(token=hf_token)

pipeline = pipeline(task="image-classification", model="cm600/fashion_classifier")

st.title("Fashion Image Classification")

file_name = st.file_uploader("Upload a Fashion-MNIST image to view its top 5 predicted probabilities for each of the following categories: \
                              T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.")

if file_name is not None:
    col1, col2 = st.columns(2)

    image = Image.open(file_name)
    col1.image(image, use_column_width=True)
    predictions = pipeline(image)

    col2.header("Probabilities")
    for p in predictions:
        col2.subheader("{}: {}%".format(p['label'], round(p['score'] * 100, 1)))
