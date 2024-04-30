import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

model = load_model('age_gender_detection_model.keras')

gender_dict = {0: 'Male', 1: 'Female'}

st.title('Gender and Age Prediction')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

def preprocess_image(image):
    image = image.resize((128, 128), Image.LANCZOS).convert('L')

    image = np.array(image)

    img = image.reshape(1, 128, 128, 1)
    img = img / 255.0
    return img

def predict_gender_age(image, model, gender_dict):
    pred = model.predict(image)
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    return pred_gender, pred_age

def display_image(image):
    plt.axis('off')
    plt.imshow(image.reshape(128, 128), cmap='gray')
    plt.show()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Predicting...")

    image = preprocess_image(image)

    # Make a prediction
    pred_gender, pred_age = predict_gender_age(image, model, gender_dict)

    # Display the prediction
    st.write(f"Predicted Gender: {pred_gender}, Predicted Age: {pred_age}")