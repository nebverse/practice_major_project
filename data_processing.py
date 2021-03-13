from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.models import Model
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import os
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix
from keras.models import model_from_yaml
from tensorflow.keras.utils import to_categorical


# making data set

target = []
images = []
flat_data = []
data_dir = "photo_trial"
categories = ["aamir khan", "hrithik roshan", "shahrukh khan", "ayushmann khurrana"]

for category in categories:
    class_num = categories.index(category)      # lable encoding
    path = os.path.join(data_dir,category)      # create path to use all the images
    for img in os.listdir(path):
        img_array = imread(os.path.join(path,img))
        # print(img_array.shape)
        img_resized = resize(img_array,(500,500,3))     # normalizes the value automaticaly
        flat_data.append(img_resized.flatten())
        images.append(img_resized)
        target.append(class_num)

flat_data = np.array(flat_data)
images = np.array(images)
target = np.array(target)


unique,count = np.unique(target,return_counts=True)


# split data
x_train,x_test,y_train,y_test = train_test_split(images,target,test_size=0.3,random_state=101)
y_train = to_categorical(y_train, 4)
y_test = to_categorical(y_test, 4)
# for i in y_train:
#     print(categories[y_train[i]])

