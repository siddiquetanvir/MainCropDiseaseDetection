import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model('crop_disease_model_final.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(180, 180))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


#Sidebar
st.sidebar.title("Crop Disease Detection")
app_mode = st.sidebar.selectbox("Select Option", ["Home", "Disease Recognition"])

# Home Page
if (app_mode == "Home"):
    st.header("WELCOME TO CROP DISEASE DETECTION APPLICATION")
    image_path = "Home_page.jpg"
    st.image(image_path, width= "stretch")
    st.markdown("""
    This application is designed to help farmers and agricultural enthusiasts identify crop diseases using advanced machine learning techniques. 
    By simply uploading an image of the affected crop, users can receive instant feedback on the potential disease affecting their plants. 
    The model has been trained on a diverse dataset of crop images, ensuring accurate and reliable predictions.
    

    ### How to Use:
    1. Navigate to the **Disease Recognition** section using the sidebar.
    2. **Upload** a clear image of the crop you want to analyze.
    3. Click the **Analyze** button to identify the disease.

    ### Note:
    - This application has a validation accuracy of **93.5%**
    - The model is trained on a dataset of **Corn, Potato, Rice, Sugarcane, and Wheat**.
    - The model can identify various diseases affecting these crops, as well as healthy plants.
    - We can expand the model to include more crops and diseases in the future.

    """)

elif (app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload an Image:")
    
    if test_image is not None:
        st.header("Disease Recognition")
        if(st.button("Show Image")):
            st.image(test_image, width="stretch")

        if(st.button("Analyze")):
            st.write("Analyzing the uploaded image...")
            result_index = model_prediction(test_image)

            class_labels = ['Corn___Common_Rust',
 'Corn___Gray_Leaf_Spot',
 'Corn___Healthy',
 'Corn___Northern_Leaf_Blight',
 'Potato___Early_Blight',
 'Potato___Healthy',
 'Potato___Late_Blight',
 'Rice___Brown_Spot',
 'Rice___Healthy',
 'Rice___Leaf_Blast',
 'Rice___Neck_Blast',
 'Sugarcane_Bacterial Blight',
 'Sugarcane_Healthy',
 'Sugarcane_Red Rot',
 'Wheat___Brown_Rust',
 'Wheat___Healthy',
 'Wheat___Yellow_Rust']

            st.success(f"Predicted disease :  {class_labels[result_index]}")