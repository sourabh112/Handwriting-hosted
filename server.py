from PIL import Image
import streamlit as st
import cv2
import numpy as np
from keras import backend as K
from keras.models import model_from_json

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 
num_of_characters = len(alphabets) + 1 
num_of_timestamps = 64 

def preprocess(img):
    (h, w) = img.shape
    final_img = np.ones([64, 256])*255  
    if w > 256:
        img = img[:, :256]
    if h > 64:
        img = img[:64, :]
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

def load_image(image_file):
	img = Image.open(image_file)
	return img

st.title("Hand-Writing Recognition")
st.write("Instructions:\n 1. Upload image of the handwritten image\n2. Image should be in size 128X32 otherwise extra part would be cropped\n3. Sample Images: [Click here](https://drive.google.com/drive/folders/1uxXmVCS0JLE972CaLG_Y6ydhksXg6iua?usp=sharing)\n4. After Uploading Image you might have to wait for a minute to Extracted Text to appear.")
st.subheader("Upload a image with handwritten text")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:
    file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
    st.write(file_details)
    st.image(load_image(image_file),width=250)
    new_model = model_from_json(open('trying.json').read())
    new_model.load_weights('model.hdf5')
    img_array = np.array(load_image(image_file))
    cv2.imwrite('static.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    image = cv2.imread("static.jpg", cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    pred = new_model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], greedy=True)[0][0])
    print("Extracted Text: "+num_to_label(decoded[0]))
    st.subheader(num_to_label(decoded[0]))

st.write("\n\n\n\n Made By:\nSourabh : 2019UCS2025 \nAditya Divyang : 2019UCS2012")