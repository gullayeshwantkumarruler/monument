# -*- coding: utf-8 -*-
"""
Created on 3-11-2021
@author: Greatlearning
"""

# -*- coding: utf-8 -*-
"""
Created on 3-11-2021
@author: Greatlearning
"""
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO
st.set_option('deprecation.showfileUploaderEncoding', False) # to avoid warnings while uploading files

# Here we will use st.cache so that we would load the model only once and store it in the cache memory which will avoid re-loading of model again and again.
# @st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('my_model4.hdf5')
  return model

# load and store the model
with st.spinner('Model is being loaded..'):
  model=load_model()

# Function for prediction
def import_and_predict(image_data, model):
    size = (128,128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction
def main():
    st.title("Monument Image Classifier")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Monument Image Classifier App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])
    class_names=['Angkor_wat','Buckingham_Palace','Burj_khalifa','Christ_the_Redeemer','Gateway_of_India','Niagara_Falls','Tajmahal','The_Eiffel_Tower','The_Great_Wall_of_China','The_Sydney_Opera_House']
    result=""
    if st.button("Predict"):
#         if file is None:
#           st.text("Please upload an image file")
#         else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image,model)
        score = tf.nn.softmax(predictions[0])
        result=class_names[np.argmax(score)
    st.success('The output is {}'.format(result))
    if st.button("About"):
       st.text("Lets LEarn")
       st.text("Built with Streamlit")

if __name__=='__main__':
    main()
