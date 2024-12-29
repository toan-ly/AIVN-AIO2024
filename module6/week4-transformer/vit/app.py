import os
import streamlit as st
from PIL import Image
from transformers import pipeline

classifier = pipeline(
    "image-classification", 
    model="thainq107/flowers-vit-base-patch16-224-in21k"
)

def main():
    st.title('Flower Classfication')
    st.subheader('Model: ViT. Dataset: Flower Dataset')
    option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Run Example Image'))
    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image", type=["jpg", "png"])
        if file is not None:
          image = Image.open(file)
          result = classifier(image)[0]
          label = result['label']
          score = result['score']*100
          st.image(image)
          st.success(f"The image is of the {label} with {score:.2f} % probability.") 

    elif option == "Run Example Image":
      image = Image.open('rose_demo.jpg')
      result = classifier('rose_demo.jpg')[0]
      label = result['label']
      score = result['score']*100
      st.image(image)
      st.success(f"The image is of the {label} with {score:.2f} % probability.") 

if __name__ == '__main__':
    main() 
