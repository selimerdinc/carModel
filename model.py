import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras.models import load_model
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob

print(tf.__version__)

import os

def load_images_from_folder(folder, m_dict, m_class):
    for filename in os.listdir(folder):
        full_path = os.path.join(folder,filename)
        if full_path.split(".")[-1] == "jpg" or full_path.split(".")[-1] == "jpeg":
            if full_path not in m_dict:
                m_dict[full_path] = m_class
    return m_dict

def classify(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_batch = np.expand_dims(img_array, axis=(0))

    img_preprocessed = preprocess_input(img_batch)
    model = load_model('color_model.h5')

    start = datetime.datetime.now()
    prediction = model.predict(img_preprocessed)
    end = datetime.datetime.now()
    pred_time=(end-start).total_seconds() * 1000
    classes = ["Black", "Blue", "Green","Red","White And Gray","Yellow"]
    m_label = classes[np.argmax((prediction))]
    print("preds: {}, \nColor ==> {}".format(prediction, m_label))
    print("Elapsed for Prediction time :  ",pred_time,"ms")    
    return m_label
    
# read images from directory
# get class names as pre defined target source (hedef sınıf sonucu)
# classify each of them and return class_name for current input image
# calculate average accuracy for input images for selected directory

    
m_dict = {}

m_dict = load_images_from_folder("val\\Black", m_dict, "Black")
m_dict=load_images_from_folder("short\\Red",m_dict,"Red")
m_dict=load_images_from_folder("short\\Green",m_dict,"Green")
m_dict=load_images_from_folder("short\\White And Gray",m_dict,"White And Gray")
m_dict=load_images_from_folder("short\\Yellow",m_dict,"Yellow")

for img in m_dict:
    print("m_path: {} => class: {}".format(img, m_dict[img]))


"""red=load_images_from_folder("short\\Red")
green=load_images_from_folder("short\\Green")
wgray=load_images_from_folder("short\\White And Gray")
blue=load_images_from_folder("short\\Blue")
yellow=load_images_from_folder("short\\Yellow")"""


correct_found = 0
for img in m_dict:
    cur_label = classify(img)
    if cur_label == m_dict[img]:
        correct_found += 1

print("total_img_ct: {}, correct_found: {}, avg_acc: {}".format(len(m_dict), correct_found, (correct_found / float(len(m_dict)))))


