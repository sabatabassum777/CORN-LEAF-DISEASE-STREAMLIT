import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64

# Load your trained model
MODEL_PATH = 'model51_vgg19.h5'
model = load_model(MODEL_PATH)

# Function to preprocess and predict
def model_predict(img_data, model):
    img_data = base64.b64decode(img_data)
    with open('image.png', 'wb') as f:
        f.write(img_data)
    img = image.load_img('image.png', target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x * 1.0 / 255
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    return preds

st.title('Corn Leaf Disease Prediction')

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img_data = uploaded_file.getvalue()
    preds = model_predict(img_data, model)
    if preds == 0:
        prediction = 'Corn Blight'
    elif preds == 1:
        prediction = 'Corn Gray Spot'
    elif preds == 2:
        prediction = 'Corn Rust'
    elif preds == 3:
        prediction = 'Corn Healthy'
    else:
        prediction = 'Unknown'

    st.write('Prediction:', prediction)
    st.image(img, caption='Uploaded Image', use_column_width=True)
