import streamlit as st
import tensorflow as tf
import cv2
import os
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
        
        train_path_Angkor_wat=[]
        train_path_Buckingham_Palace=[]
        train_path_Burj_khalifa=[]
        train_path_Christ_the_Redeemer=[]
        train_path_Gateway_of_India=[]
        train_path_Niagara_Falls=[]
        train_path_Tajmahal=[]
        train_path_The_Eiffel_Tower=[]
        train_path_The_Great_Wall_of_China=[]
        train_path_The_Sydney_Opera_House=[]
        
        for folder_name in ['Angkor_wat/','Buckingham_Palace/','Burj_khalifa/','Christ_the_Redeemer/','Gateway_of_India/','Niagara_Falls/','Tajmahal/','The_Eiffel_Tower/','The_Great_Wall_of_China/','The_Sydney_Opera_House/']:
    
          #Path of the folder
          images_path = os.listdir(folder_name)

          for i, image_name in enumerate(images_path): 
            if folder_name=='Angkor_wat/':
                train_path_Angkor_wat.append(folder_name+image_name)
            elif folder_name=='Buckingham_Palace/':
                train_path_Buckingham_Palace.append(folder_name+image_name)
            elif folder_name=='Burj_khalifa/':
                train_path_Burj_khalifa.append(folder_name+image_name)
            elif folder_name=='Christ_the_Redeemer/':
                train_path_Christ_the_Redeemer.append(folder_name+image_name)
            elif folder_name=='Gateway_of_India/':
                train_path_Gateway_of_India.append(folder_name+image_name)
            elif folder_name=='Niagara_Falls/':
                train_path_Niagara_Falls.append(folder_name+image_name)
            elif folder_name=='Tajmahal/':
                train_path_Tajmahal.append(folder_name+image_name)
            elif folder_name=='The_Eiffel_Tower/':
                train_path_The_Eiffel_Tower.append(folder_name+image_name)
            elif folder_name=='The_Great_Wall_of_China/':
                train_path_The_Great_Wall_of_China.append(folder_name+image_name)
            elif folder_name=='The_Sydney_Opera_House/':
                train_path_The_Sydney_Opera_House.append(folder_name+image_name)

        Angkor_wat=[]
        Buckingham_Palace=[]
        Burj_khalifa=[]
        Christ_the_Redeemer=[]
        Gateway_of_India=[]
        Niagara_Falls=[]
        Tajmahal=[]
        The_Eiffel_Tower=[]
        The_Great_Wall_of_China=[]
        The_Sydney_Opera_House=[]
        
        for i in train_path_Angkor_wat:
          image = Image.open(i).resize((200, 200))
          Angkor_wat.append(image)
        for i in train_path_Buckingham_Palace:
          image = Image.open(i).resize((200, 200))
          Buckingham_Palace.append(image)
        for i in train_path_Burj_khalifa:
          image = Image.open(i).resize((200, 200))
          Burj_khalifa.append(image)
        for i in train_path_Christ_the_Redeemer:
          image = Image.open(i).resize((200, 200))
          Christ_the_Redeemer.append(image)
        for i in train_path_Gateway_of_India:
          image = Image.open(i).resize((200, 200))
          Gateway_of_India.append(image)
        for i in train_path_Niagara_Falls:
          image = Image.open(i).resize((200, 200))
          Niagara_Falls.append(image)
        for i in train_path_Tajmahal:
          image = Image.open(i).resize((200, 200))
          Tajmahal.append(image)
        for i in train_path_The_Eiffel_Tower:
          image = Image.open(i).resize((200, 200))
          The_Eiffel_Tower.append(image)
        for i in train_path_The_Great_Wall_of_China:
          image = Image.open(i).resize((200, 200))
          The_Great_Wall_of_China.append(image)
        for i in train_path_The_Sydney_Opera_House:
          image = Image.open(i).resize((200, 200))
          The_Sydney_Opera_House.append(image)
        
        if st.button("show_similar_trained_images"):
          
          with st.sidebar:
              st.header("Configuration")
              with st.form(key="grid_reset"):
                  n_photos = st.slider("Number of cat photos:", 4, 128, 16)
                  n_cols = st.number_input("Number of columns", 2, 8, 4)
                  st.form_submit_button(label="Reset images and layout")

              if st.button("About"):
                 st.text("Lets Learn building a Monument image classifiier")
                 st.text("Deploying with Streamlit")

          if result=='Angkor_wat':
            final_images =Angkor_wat

          elif result=='Buckingham_Palace':
            final_images =Buckingham_Palace

          elif result=='Burj_khalifa':
            final_images =Burj_khalifa

          elif result=='Christ_the_Redeemer':
            final_images =Christ_the_Redeemer

          elif result=='Gateway_of_India':
            final_images =Gateway_of_India

          elif result=='Niagara_Falls':
            final_images =Niagara_Falls

          elif result=='Tajmahal':
            final_images =Tajmahal

          elif result=='The_Eiffel_Tower':
            final_images =The_Eiffel_Tower

          elif result=='The_Great_Wall_of_China':
            final_images =The_Great_Wall_of_China

          elif result=='The_Sydney_Opera_House':
            final_images =The_Sydney_Opera_House
            
          n_rows = 1 + len(final_images) // int(n_cols)
          rows = [st.container() for _ in range(n_rows)]
          cols_per_row = [r.columns(n_cols) for r in rows]
          cols = [column for row in cols_per_row for column in row]

          for image_index, mon_image in enumerate(final_images):
              cols[image_index].image(mon_image)


          
       
  


if __name__=='__main__':
    main()
