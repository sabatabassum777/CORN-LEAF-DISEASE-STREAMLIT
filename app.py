import streamlit as st
import base64
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
MODEL_PATH = 'model51_vgg19.h5'
model = load_model(MODEL_PATH)

def model_predict(img):
    img = image.load_img(img, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x * 1.0 / 255
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    return preds

def show_prediction(preds):
    if preds == 0:
        st.write('Prediction: Corn Blight')
        st.write('Management: Spacing, Drip Irrigation, Proper Pruning')
        st.write('Pesticides: Mancozeb, propiconazole, azoxystrobin')
        st.write('Product Link: [Mancozeb on Amazon](https://www.amazon.in/AD-45-Mancozeb-75-WP-Fungicide/dp/B07JDSJY6C)')
    elif preds == 1:
        st.write('Prediction: Corn Gray Spot')
        st.write('Management: Planting Density, Spacing, Drip Irrigation')
        st.write('Pesticides: Mancozeb, propiconazole, azoxystrobin')
        st.write('Product Link: [Propiconazole on Amazon](https://www.amazon.in/PROSPELL-250-AZOXYSTROBIN-MANCOZEB-Fungicide/dp/B07MFH9C1C)')
    elif preds == 2:
        st.write('Prediction: Corn Rust')
        st.write('Management: Cutting shoot tips, Spacing, Drip Irrigation')
        st.write('Pesticides: Propiconazole, Tebuconazole')
        st.write('Product Link: [Folicur on BigHaat](https://www.bighaat.com/products/folicur)')
    elif preds == 3:
        st.write('Prediction: Corn Healthy')
        st.write('No measures needed.')
        st.write('No pesticide needed.')
        st.write('Product Link: [Ugaoo Organic Vermicompost on Amazon](https://www.amazon.in/Ugaoo-Organic-Vermicompost-Fertilizer-Plants/dp/B0BDVN579S/ref=sr_1_3_sspa?hvadid=82944601526132&hvbmt=bp&hvdev=c&hvqmt=p&keywords=organic+fertilizer+for+plants&qid=1696276146&sr=8-3-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1)')
    else:
        st.write('Prediction: Unknown')
        st.write('Unknown disease.')
        st.write('Unknown pesticides.')
        st.write('Unknown product link.')

def main():
    st.title('Corn Leaf Disease Prediction')
    st.sidebar.title('Upload Image')
    uploaded_file = st.sidebar.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write('')
        st.write('Classifying...')
        preds = model_predict(uploaded_file)
        show_prediction(preds)

if __name__ == '__main__':
    main()
