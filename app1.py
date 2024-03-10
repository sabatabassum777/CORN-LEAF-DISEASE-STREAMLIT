import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

st.set_page_config(page_title='Corn Leaf Disease Prediction')

MODEL_PATH = 'model51_vgg19.h5'
MODEL_WITH_OPTIMIZER_PATH = 'model_with_optimizer.h5'

# Check if the model with optimizer state already exists
if not os.path.exists(MODEL_WITH_OPTIMIZER_PATH):
    # Load the original model
    model = load_model(MODEL_PATH)
    # Save the model with optimizer state included
    model.save(MODEL_WITH_OPTIMIZER_PATH, include_optimizer=True)

# Load the model with optimizer state
model = load_model(MODEL_WITH_OPTIMIZER_PATH)

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

# HTML code for the camera capture
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Corn Leaf Disease Prediction</title>
    <!-- CSS only -->
    <link rel="stylesheet" href="../static/css/style.css">
</head>
<body>
    <div class="card">
        <div class="logo-container">
        </div>
        <div class="line"></div>
        <div class="text">Examine your crop here</div>
        <button id="start-camera" class="btn"><i class="animation"></i>Capture<i class="animation"></i></button>
        <video id="video" width="320" height="240" autoplay></video>
        <button id="click-photo">Click Photo</button>
        <form action="/predict" method="post" >
        <div id="dataurl-container">
            <canvas id="canvas" width="320" height="240"></canvas>
            <div id="dataurl-header">Image Data URL</div>
            <input style="display:none" type="text" id="dataurl" name="sec" readonly>
            <button type="submit">Submit</button>
        </div>
        </form>
        <button>Disease {{ preds }}</button>
        <button>Measures {{ mes }}</button>
        <button>Pesticides {{ pest }}</button>
        <a href="{{ link }}"><button class="btn1">Link {{ link }}</button></a>
        </div>
    <script>

        let camera_button = document.querySelector("#start-camera");
        let video = document.querySelector("#video");
        let click_button = document.querySelector("#click-photo");
        let canvas = document.querySelector("#canvas");
        let dataurl = document.querySelector("#dataurl");
        let dataurl_container = document.querySelector("#dataurl-container");

        camera_button.addEventListener('click', async function() {
            let stream = null;

            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            }
            catch(error) {
                alert(error.message);
                return;
            }

            video.srcObject = stream;

            video.style.display = 'block';
            camera_button.style.display = 'none';
            click_button.style.display = 'block';
        });

        click_button.addEventListener('click', function() {
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            let image_data_url = canvas.toDataURL('image/jpeg');

            dataurl.value = image_data_url;
            dataurl_container.style.display = 'block';
        });

    </script>
</body>
</html>
"""

# Streamlit app code
st.write(html_code, unsafe_allow_html=True)
