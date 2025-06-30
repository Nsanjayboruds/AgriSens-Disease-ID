import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Hide Streamlit footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Function to predict plant disease
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("🌾 SmartFarm")
app_mode = st.sidebar.selectbox("Navigate", ["🏠 Home", "🦠 Disease Recognition"])

# Display banner image
img = Image.open("Diseases.png")
st.image(img)

# Home Page
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>🌿 SmartFarm Disease Detection</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 👤 Reviewer") 
    st.markdown("**Name**: Nishant Borude")  
    st.markdown("**Location**: India")  
    st.markdown("**Status**: Active profile")

# Disease Prediction Page
elif app_mode == "🦠 Disease Recognition":
    st.header("📸 Upload Plant Image")
    test_image = st.file_uploader("Choose an Image:")

    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    if test_image and st.button("Predict"):
        st.snow()
        st.write("🔍 Predicting disease...")
        result_index = model_prediction(test_image)

        # Class Labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        st.success(f"🌱 Model Prediction: **{class_name[result_index]}**")
