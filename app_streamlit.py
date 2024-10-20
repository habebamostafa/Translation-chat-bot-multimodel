import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import fitz
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from keras import backend as K
import json
import requests
import logging
logging.basicConfig(level=logging.DEBUG)

# Global Variables
image_to_text_model = None
translation_model = None

# Load Chatbot Model
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
chat_model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to Get Chatbot Response
def get_chatbot_response(user_input):
    try:
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        response_ids = chat_model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response
    except Exception as e:
        st.error(f"Error in generating chatbot response: {str(e)}")
        return None

# Function to Read PDF
def read_pdf(file):
    try:
        text = ""
        with fitz.open(stream=file.read(), filetype='pdf') as pdf_document:
            for page in pdf_document:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

# Load Image-to-Text Model
def load_image_to_text_model():
    global image_to_text_model
    if image_to_text_model is None:
        try:
            image_to_text_model = tf.keras.models.load_model('cnn_model.h5')
            st.success("Image-to-text model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load the image-to-text model: {str(e)}")
    return image_to_text_model

# Load Translation Model
def load_translation_model():
    global translation_model
    if translation_model is None:
        try:
            translation_model = pipeline("translation_en_to_ar", model='Helsinki-NLP/opus-mt-en-ar')
            st.success("Translation model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load the translation model: {str(e)}")
    return translation_model

# Function for Image-to-Text Conversion
def image_to_text(image):
    model = load_image_to_text_model()
    if model is None:
        return None
    try:
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        resized_image = cv2.resize(gray_image, (32, 32)) / 255.0
        input_image = np.expand_dims(resized_image, axis=-1)  # Shape (32, 32, 1)
        input_image = np.expand_dims(input_image, axis=0)  # Shape (1, 32, 32, 1)
        input_image_rgb = np.repeat(input_image, 3, axis=-1)  # Shape (1, 32, 32, 3)

        prediction = model.predict(input_image_rgb)
        return prediction
    except Exception as e:
        st.error(f"Error in image-to-text conversion: {str(e)}")
        return None

# Function for Text Translation
def translate_sentence(english_sentence):
    model = load_translation_model()
    if model is None:
        return None
    try:
        translated = model(english_sentence)
        return translated[0]['translation_text']
    except Exception as e:
        st.error(f"Error in translation: {str(e)}")
        return None

# Streamlit UI
st.title("Multimodal Translation and Chatbot App")

# App Option
option = st.selectbox("Choose Input Type", ["Text Translation", "Chat", "Image to Text", "PDF Translation"])

if option == "Text Translation":
    input_text = st.text_input("Enter text for translation")
    if st.button("Translate"):
        translated_text = translate_sentence(input_text)
        if translated_text:
            st.write("Translated Text:", translated_text)

elif option == "Chat":
    user_input = st.text_input("You:")
    if st.button("Chat"):
        if user_input:
            bot_response = get_chatbot_response(user_input)
            if bot_response:
                st.text(f"Bot: {bot_response}")

elif option == "Image to Text":
    upload_option = st.radio("Choose Input Method", ("Try Sample Image", "Upload Image"))

    predefined_images = {
        "Sample Image 1": "Screenshot 2024-10-06 161450.png",
        "Sample Image 2": "Screenshot 2024-10-06 161505.png",
    }

    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Extract Text"):
                extracted_text = image_to_text(image)
                st.write("Extracted Text:", extracted_text)

    elif upload_option == "Try Sample Image":
        selected_image = st.selectbox("Select a sample image", list(predefined_images.keys()))
        image_path = predefined_images[selected_image]
        image = Image.open(image_path)
        st.image(image, caption="Sample Image", use_column_width=True)
        if st.button("Extract Text"):
            extracted_text = image_to_text(image)
            st.write("Extracted Text:", extracted_text)

elif option == "PDF Translation":
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file is not None:
        pdf_text = read_pdf(pdf_file)
        if pdf_text:
            st.write("Extracted Text from PDF:")
            st.write(pdf_text)

            if st.button("Translate PDF Text"):
                translated_pdf_text = translate_sentence(pdf_text)
                st.write("Translated PDF Text:", translated_pdf_text)
