from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import os
from keras.models import model_from_json
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

@st.cache()

def load_model():
    # model = keras.models.load_model('major_model.hdf5')

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

st.title("MAJOR PROJECT")
upload = st.sidebar.file_uploader(label="UPLOAD IMAGE HERE")

if upload is not None:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    img = Image.open(upload)
    st.image(img, caption='Uploaded Image', width=300)
    model = load_model()

    if st.sidebar.button('PREDICT'):
        st.sidebar.write("Result:")
        x = cv2.resize(opencv_image, (500, 500))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        y = model.predict(x)
        # label = decode_predictions(y)
        category = ["aamir khan", "hrithik roshan", "shahrukh khan", "ayushmann khurrana"]
        my_list = [y[0][0],y[0][1],y[0][2],y[0][3]]
        if max(my_list) == y[0][0]:
            st.sidebar.subheader(category[0])

        elif max(my_list) == y[0][1]:
            st.sidebar.subheader(category[1])

        elif max(my_list) == y[0][2]:
            st.sidebar.subheader(category[2])

        elif max(my_list) == y[0][3]:
            st.sidebar.subheader(category[3])


# summary of project

st.write("This is an image classifier with four classes that is category = [aamir khan, hrithik roshan, shahrukh khan, ayushmann khurrana]."
         "I am using Transfer Learning (InceptionV3) here giving my own dataset downloaded with the help of bing downloader library,"
         "and forming it into training and testing datasets."
         "You can predict within the category given above by uploading an image and clicking on predict.")

st.subheader("I am taking 200 images of each actor for training and testing.")