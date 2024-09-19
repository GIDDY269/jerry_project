import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import numpy as np
import cv2
import imutils

# Load your trained model
model = load_model('model\\20240916-144821-Brain_tumor_model.keras')

def crop_imgs(img, add_pixels_value=0, img_size=(100, 100)):
    """
    Finds the extreme points on the image, crops the rectangular out of them,
    and resizes the cropped image to a consistent size.
    """
    # Convert the PIL image to a NumPy array
    img = np.array(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold the image, then perform erosions and dilations to remove noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # Find contours in thresholded image and grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = add_pixels_value
    new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS, extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()
    # Resize the cropped image to the specified size
    new_img = cv2.resize(new_img, img_size)
        

    return new_img

def preprocess_imgs(img, img_size):
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img

# Create the Streamlit app
st.title("Brain Tumor Detection App")

st.write("""
### Upload an MRI scan of the brain, and the model will predict whether a tumor is present or not, along with the probability.
""")

# File upload input
uploaded_file = st.file_uploader("Choose an MRI image...", type=['png','jpg','jpeg','jfif'])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI scan', use_column_width=True)

    
    cropped_img = crop_imgs(image)
    # Preprocess the image
    processed_image = preprocess_imgs(cropped_img,img_size=[224,224])

    # Make a prediction
    prediction = model.predict(processed_image)
    print(prediction)
    probability = float(prediction[0])  # Get the probability score

    # Determine the result based on a threshold (0.5 for binary classification)
    result = "Tumor Detected" if probability > 0.5 else "No Tumor Detected"

    # Display the prediction and probability
    st.write(f"Prediction: **{result}**")
    st.write(f"Probability of Tumor: **{probability * 100:.2f}%**")
