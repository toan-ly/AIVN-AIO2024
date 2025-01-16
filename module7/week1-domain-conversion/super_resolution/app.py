import torch
import torch.nn as nn
import streamlit as st

from PIL import Image
from torchvision import transforms
import numpy as np

from unet_sr import UNet

LOW_IMG_HEIGHT = 64
LOW_IMG_WIDTH = 64

def proces_img(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    img = transforms.functional.to_tensor(img)
    img = img * 2 - 1
    img = img.type(torch.float32)
    img = img.unsqueeze(0)
    return img

unet_model = UNet()

# load model 
unet_model.load_state_dict(torch.load('SR_unet_model.pt'))
unet_model.eval()

st.title('Pytorch Super Resolution')

img = st.sidebar.selectbox(
    'Select Image',
    ("0.png",  "1.png",  "2.png",  "3.png",  "4.png",  "5.png")
)

# model= "saved_models/" + style_name + ".pth"
input_image = "imgs/input/" + img
output_image = "imgs/target/" + img

st.write('### Source image:')
image = Image.open(input_image)
st.image(image, width=200) # image: numpy array

clicked = st.button('Inferece')

if clicked:
    st.write('### SR image:')
    img = proces_img(input_image)
    sr_image = unet_model(img)
    sr_image = sr_image.squeeze(0).detach().numpy()
    sr_image = np.transpose(sr_image, (1, 2, 0))
    sr_image = Image.fromarray(((sr_image + 1) * 127.5).astype(np.uint8))
    st.image(sr_image, width=400)

    st.write('### Target image:')
    image = Image.open(output_image)
    st.image(image, width=400)