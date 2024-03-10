import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

st.set_page_config(page_title='Corn Leaf Disease Prediction')

MODEL_PATH = 'model51_vgg19.h5'
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x * 1.0 / 255
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        return 'Corn Blight', 'Spacing,Drip Irrigation,Proper Pruning', 'Mancozeb, propiconazole, azoxystrobin', 'https://www.amazon.in/AD-45-Mancozeb-75-WP-Fungicide/dp/B07JDSJY6C'
    elif preds == 1:
        return 'Corn Gray Spot', 'Planting Density,Spacing,Drip Irrigation', 'Mancozeb, propiconazole, azoxystrobin', 'https://www.amazon.in/PROSPELL-250-AZOXYSTROBIN-MANCOZEB-Fungicide/dp/B07MFH9C1C'
    elif preds == 2:
        return 'Corn Rust', 'cutting shoot tips,Spacing,Drip Irrigation', 'Propiconazole, Tebuconazole', 'https://www.bighaat.com/products/folicur'
    elif preds == 3:
        return 'Corn Healthy', 'No measures', 'No pesticide', 'https://www.amazon.in/Ugaoo-Organic-Vermicompost-Fertilizer-Plants/dp/B0BDVN579S/ref=sr_1_3_sspa?hvadid=82944601526132&hvbmt=bp&hvdev=c&hvqmt=p&keywords=organic+fertilizer+for+plants&qid=1696276146&sr=8-3-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1'
    else:
        return 'Unknown', 'Unknown', 'Unknown', 'Unknown'

st.title('Corn Leaf Disease Prediction')

camera_button = st.button('Capture Image from Camera')
if camera_button:
    st.write('Please wait while we connect to your camera...')
    webrtc_ctx = webrtc_streamer(key="example")
    if webrtc_ctx.video_transformer:
        image_data = webrtc_ctx.video_transformer.image
        if image_data is not None:
            img = Image.fromarray(image_data)
            st.image(img, caption='Captured Image.', use_column_width=True)
            img_path = 'uploads/new.png'
            img.save(img_path)
            preds, mes, pest, link = model_predict(img_path, model)
            st.write(f'Disease: {preds}')
            st.write(f'Measures: {mes}')
            st.write(f'Pesticides: {pest}')
            st.write(f'Link: [{link}]({link})')

image_file = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'])
if image_file is not None:
    img = Image.open(image_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    img_path = 'uploads/new.png'
    img.save(img_path)
    preds, mes, pest, link = model_predict(img_path, model)
    st.write(f'Disease: {preds}')
    st.write(f'Measures: {mes}')
    st.write(f'Pesticides: {pest}')
    st.write(f'Link: [{link}]({link})')
