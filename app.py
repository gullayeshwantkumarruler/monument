import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
st.set_option('deprecation.showfileUploaderEncoding', False) # to avoid warnings while uploading files

# Here we will use st.cache so that we would load the model only once and store it in the cache memory which will avoid re-loading of model again and again.
@st.cache(allow_output_mutation=True)
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
      if file is None:
        st.write("please upload an image")
      else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image,model)
        score = tf.nn.softmax(predictions[0])
        result= class_names[np.argmax(score)]
        st.write('This is {}'.format(result))
        st.caption("The result is trained on similar images like: ")
        if result=='Angkor_wat':
          images=["Angkor_wat/Angkor_wat (1).jpg","Angkor_wat/Angkor_wat (2).jpg","Angkor_wat/Angkor_wat (3).jpg","Angkor_wat/Angkor_wat (4).jpg"]
          for i in images:
            image = Image.open(i).resize((100, 100))
            st.image(image, caption='Angkor_wat')
        elif result=='Buckingham_Palace':
          images=["Buckingham_Palace/Buckingham_Palace (1).jpg","Buckingham_Palace/Buckingham_Palace (2).jpg","Buckingham_Palace/Buckingham_Palace (3).jpg","Buckingham_Palace/Buckingham_Palace (4).jpg"]
          for i in images:
            image = Image.open(i).resize((100, 100))
            st.image(image, caption='Buckingham_Palace')
        elif result=='Burj_khalifa':
          images=["Burj_khalifa/Burj_khalifa (1).jpg","Burj_khalifa/Burj_khalifa (2).jpg","Burj_khalifa/Burj_khalifa (3).jpg","Burj_khalifa/Burj_khalifa (4).jpg"]
          for i in images:
            image = Image.open(i).resize((100, 100))
            st.image(image, caption='Burj_khalifa')
        elif result=='Christ_the_Redeemer':
          images=["Christ_the_Redeemer/Christ_the_Redeemer (1).jpg","Christ_the_Redeemer/Christ_the_Redeemer (2).jpg","Christ_the_Redeemer/Christ_the_Redeemer (3).jpg","Christ_the_Redeemer/Christ_the_Redeemer (4).jpg"]
          for i in images:
            image = Image.open(i).resize((100, 100))
            st.image(image, caption='Christ_the_Redeemer')
        elif result=='Gateway_of_India':
          images=["Gateway_of_India/Gateway_of_India (1).jpg","Gateway_of_India/Gateway_of_India (2).jpg","Gateway_of_India/Gateway_of_India (3).jpg","Gateway_of_India/Gateway_of_India (4).jpg"]
          for i in images:
            image = Image.open(i).resize((100, 100))
            st.image(image, caption='Gateway_of_India')
        elif result=='Niagara_Falls':
          images=["Niagara_Falls/Niagara_Falls (1).jpg","Niagara_Falls/Niagara_Falls (2).jpg","Niagara_Falls/Niagara_Falls (3).jpg","Niagara_Falls/Niagara_Falls (4).jpg"]
          for i in images:
            image = Image.open(i).resize((100, 100))
            st.image(image, caption='Niagara_Falls')
        elif result=='Tajmahal':
          images=["Tajmahal/Tajmahal (1).jpg","Tajmahal/Tajmahal (2).jpg","Tajmahal/Tajmahal (3).jpg","Tajmahal/Tajmahal (4).jpg"]
          for i in images:
            image = Image.open(i).resize((100, 100))
            st.image(image, caption='Tajmahal')
        elif result=='The_Eiffel_Tower':
          images=["The_Eiffel_Tower/The_Eiffel_Tower (1).jpg","The_Eiffel_Tower/The_Eiffel_Tower (2).jpg","The_Eiffel_Tower/The_Eiffel_Tower (3).jpg","The_Eiffel_Tower/The_Eiffel_Tower (4).jpg"]
          for i in images:
            image = Image.open(i).resize((100, 100))
            st.image(image, caption='The_Eiffel_Tower')
        elif result=='The_Great_Wall_of_China':
          images=["The_Great_Wall_of_China/The_Great_Wall_of_China (1).jpg","The_Great_Wall_of_China/The_Great_Wall_of_China (2).jpg","The_Great_Wall_of_China/The_Great_Wall_of_China (3).jpg","The_Great_Wall_of_China/The_Great_Wall_of_China (4).jpg"]
          for i in images:
            image = Image.open(i).resize((100, 100))
            st.image(image, caption='The_Great_Wall_of_China')
        elif result=='The_Sydney_Opera_House':
          images=["The_Sydney_Opera_House/The_Sydney_Opera_House (1).jpg","The_Sydney_Opera_House/The_Sydney_Opera_House (2).jpg","The_Sydney_Opera_House/The_Sydney_Opera_House (3).jpg","The_Sydney_Opera_House/The_Sydney_Opera_House (4).jpg"]
          for i in images:
            image = Image.open(i).resize((100, 100))
            st.image(image, caption='The_Sydney_Opera_House')

          
       
  
    if st.button("About"):
       st.text("Lets Learn building a Monument image classifiier")
       st.text("Deploying with Streamlit")

if __name__=='__main__':
    main()
