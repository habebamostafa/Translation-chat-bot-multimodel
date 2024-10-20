import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import fitz  # PyMuPDF
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
# Helsinki-NLP/opus-mt-en-ar
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

mapping_inverse = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'}

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
def convert_2_gray(image):
    image_np = np.array(image)
    if len(image_np.shape) < 2:
        raise ValueError("Input image is not valid. It should have at least 2 dimensions.")

    if len(image_np.shape) == 3:
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    elif len(image_np.shape) == 2:
        gray_image = image_np
    else:
        raise ValueError("Unexpected image format.")

    return gray_image

def binarization(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return thresh

def dilate(image, words=False):
    m, n = 3, 1
    if words:
        m = n = 6
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, m))
    return cv2.dilate(image, rect_kernel, iterations=3)

def find_rect(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    return sorted(rects, key=lambda x: x[0])

def extract(image):
    model = load_image_to_text_model()  # Load the image-to-text model
    chars = []
    image_cpy = convert_2_gray(image)
    bin_img = binarization(image_cpy)
    full_dil_img = dilate(bin_img, words=True)
    words = find_rect(full_dil_img)

    prev_x_end = 0
    for word in words:
        x, y, w, h = word
        img = image_cpy[y:y + h, x:x + w]
        if x - prev_x_end > 20:
            chars.append(' ')
        prev_x_end = x + w

        bin_img = binarization(convert_2_gray(img))
        dil_img = dilate(bin_img)
        char_parts = find_rect(dil_img)

        for char in char_parts:
            cx, cy, cw, ch = char
            ch_img = img[cy:cy + ch, cx:cx + cw]
            empty_img = np.full((32, 32, 1), 255, dtype=np.uint8)
            resized = cv2.resize(ch_img, (16, 22), interpolation=cv2.INTER_CUBIC)
            gray = convert_2_gray(resized)
            empty_img[3:3 + 22, 3:3 + 16, 0] = gray

            gray_rgb = cv2.cvtColor(empty_img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            prediction = model.predict(np.array([gray_rgb]), verbose=0)
            predicted_char = mapping_inverse[np.argmax(prediction)]
            chars.append(predicted_char)

    return ''.join(chars)

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
                extracted_text = extract(image)
                st.write("Extracted Text:", extracted_text)

    elif upload_option == "Try Sample Image":
        selected_image = st.selectbox("Select a sample image", list(predefined_images.keys()))
        image_path = predefined_images[selected_image]
        image = Image.open(image_path)
        st.image(image, caption="Sample Image", use_column_width=True)
        if st.button("Extract Text"):
            extracted_text = extract(image)
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
