import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import fitz
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from transformers import pipeline
import torch
import sentencepiece as spm
from dotenv import load_dotenv
load_dotenv()
from keras import backend as K
K.clear_session()

# Initialize session state for chat history if not already done
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

def read_pdf(file):
    text = ""
    # Use BytesIO to read the uploaded file
    with fitz.open(stream=file.read(), filetype='pdf') as pdf_document:
        for page in pdf_document:
            text += page.get_text()
    return text

def load_image_to_text_model():
    return torch.load('cnn_model.h5')

def load_translation_model():
    return tf.keras.models.load_model('model')


def translate_sentence(english_sentence):
    model = pipeline("translation_en_to_ar", model='model')
    translate_sentence = model(english_sentence)
    translated = translate_sentence[0]['translation_text']
    return translated


def image_to_text(image, model):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(gray_image, (32, 32)) / 255.0
    input_image = np.expand_dims(resized_image, axis=-1)  # Shape is (32, 32, 1)
    input_image = np.expand_dims(input_image, axis=0)  # Shape is (1, 32, 32, 1)

    # If your model expects 3 channels, you can repeat the gray image across the 3rd dimension
    input_image_rgb = np.repeat(input_image, 3, axis=-1)  # Shape becomes (1, 32, 32, 3)

    prediction = model.predict(input_image_rgb)
    return prediction

predefined_images = {
    "Sample Image 1": "Screenshot 2024-10-06 161450.png",
    "Sample Image 2": "Screenshot 2024-10-06 161505.png",
}

mapping_inverse = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'}
api_key = os.getenv('API_KEY')
AZURE_OPENAI_ENDPOINT = "https://chatbottservice.openai.azure.com/"
MODEL_NAME = "gpt-35-turbo" 

def ask_openai(prompt):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY,
    }
    
    data = {
        "messages": [{"role": "system", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.7,
        "model": MODEL_NAME,  # Your model name
    }
    json_data = json.dumps(data)

    try:
        # Send the POST request with the JSON data
        response = requests.post(
            "https://chatbottservice.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview",
            headers=headers,
            data=json_data  # Use 'data' instead of 'json' to allow Content-Length
        )
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.HTTPError as e:
        return f"Error: {e.response.status_code} - {e.response.text}"


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

image_to_text_model = load_image_to_text_model()

# Streamlit interface
st.title("Multimodal Translation and Chatbot App")

option = st.selectbox("Choose Input Type", ["Text Translation","Chat", "Image to Text", "PDF Translation"])

if option == "Text Translation":
    input_text = st.text_input("Enter text for translation")
    if st.button("Translate"):
        translated = translate_sentence(input_text)
        st.write("Translated Text:", translated)

elif option == "Chat":
    user_input = st.text_input("You:", "")

    if st.button("Chat"):
        if user_input:
            try:
                bot_response = ask_openai(user_input)
                st.text(f"Bot: {bot_response}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif option == "Image to Text":
    upload_option = st.radio("Choose Input Method", ("Try Sample Image","Upload Image"))

    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Extract Text"):
                extracted_text = extract(image)
                st.write("Extracted Text: ", extracted_text)

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
        st.write("Extracted Text from PDF:")
        st.write(pdf_text)

        if st.button("Translate PDF Text"):
            translated_pdf_text = translate_sentence(pdf_text)
            st.write("Translated PDF Text:")
            st.write(translated_pdf_text)
