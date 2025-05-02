import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
from keras.models import load_model
from skimage.transform import resize
import numpy as np

# Title and Display Image
st.title(':rainbow[HAPPY OR SAD?]')
pic = Image.open('pic.png')
st.image(pic, width=850)

# Load the model
model = load_model('model_cnn.h5')

# Define categories (make sure this is your actual list of labels)
categories = ['Happy', 'Sad']


def main():
    # File uploader
    file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

    # Check if file is uploaded
    if file is not None:
        # Display uploaded image
        show_file = st.empty()
        if isinstance(file, BytesIO):
            img = Image.open(file)
            show_file.image(img, caption='Uploaded Image', use_column_width=True)

        # Predict when the button is clicked
        pred = st.button('PREDICT')
        if pred:
            # Resize and reshape image for model prediction
            img_array = np.array(img)
            img_resized = resize(img_array, (150, 150, 3))
            img_resized = img_resized.reshape(1, 150, 150, 3)

            # Make prediction
            y_new = model.predict(img_resized)
            ind = y_new.argmax()
            prediction = categories[ind]

            # Show the result
            st.write(f"# Prediction: {prediction}")


# Run the main function
main()

