import ultralytics
from ultralytics import YOLO
import streamlit as st
import numpy as np
import PIL
from PIL import Image
import requests
from io import BytesIO
import cv2
from ultralytics.utils.plotting import Annotator


# Give the path of the best.pt (best weights)
model_path = 'best.pt'

# Setting page layout
st.set_page_config(
    page_title="Storm Damage Assessment Model",  # Setting page title
    page_icon= r"C:\Users\BY195HX\Downloads\EY logo Black 1.png",  # Setting page icon
    layout="wide",  # Setting layout to wide
    initial_sidebar_state="expanded",  # Expanding sidebar by default

)

# Creating sidebar
with st.sidebar:
    st.header("Image Config")  # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload a Satellite Image", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Damage Detection")
st.caption('Updload a photo by selecting :blue[Browse files]')
st.caption('Then click the :blue[Detect Objects] button and check the result.')
# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        image_width, image_height = uploaded_image.size
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 width=image_width
                 )

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image,
                        conf=confidence,
                        line_width=1,
                        show_labels=True,
                        show_conf=False
                        )
    boxes = res[0].boxes
    res_plotted = res[0].plot(labels=True, line_width=1)[:, :, ::-1]
    with col2:
        st.image(res_plotted,
                 caption='Detected Image',
                 width=image_width
                 )