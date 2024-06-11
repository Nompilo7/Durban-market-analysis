import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_save_path = "price_predict_model.pkl"
with open(model_save_path, 'rb') as file:
    loaded_model = pickle.load(file)

st.image("Onionbrown1.webp"
    
# Background image URL or local path
background_image_url = "Onionbrown1.webp"

# Custom CSS
background_css = f"""
<style>
    .stApp {{
        background: url("{background_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(background_css, unsafe_allow_html=True)



# Sidebar with input field descriptions
st.sidebar.header("Description of The Required Input Fields")
st.sidebar.markdown("**Province**: The provinces producing Onion brown.")
st.sidebar.markdown("**Size_Grade**: The sizes of the brown onion packages.")
st.sidebar.markdown("**Weight_Kg**: The weight of onion brown in kilogram.")
st.sidebar.markdown("**Low_Price**: The lowest price the onion brown cost in the market.")
st.sidebar.markdown("**Sales_Total**: The total price purchase onion brown.")
st.sidebar.markdown("**Stock_On_Hand**: The onion brown stock currently available in the warehouse.")


# Streamlit interface
st.title("Onion Brown Average Price Prediction")
# st.image("https://lh3.googleusercontent.com/p/AF1QipOcO8yiWIR3cXIW8QIRkyrTTIqdCrgdT8FimUA9=s1360-w1360-h1020")

# Function to preprocess user inputs and make predictions
def predict_price(Province,Size_Grade,Weight_Kg,Low_Price,Sales_Total,Stock_On_Hand,month,day):
    # Assuming label encoding mappings are known
    province_mapping = {'NORTHERN CAPE':2, 'WESTERN CAPE - CERES':7, 'WEST COAST':6,'SOUTH WESTERN FREE STATE':4, 'WESTERN FREESTATE':8, 'NATAL':1,'KWAZULU NATAL':0,
                        'OTHER AREAS':3, 'TRANSVAAL':5} 
   # Replace with actual mappings
    size_grade_mapping = {'1M':1, '2L':6, '1R':2, '1L':0, '1Z':5, '1S':3, '1X':4, '3L':11, '2R':8, '2M':7, '3S':14,
       '3Z':15, '3M':12, '2Z':10, '3R':13, '2S':9}
    # Convert categorical inputs to numerical using label encoding
    province_encoded = province_mapping.get(Province,-1)  # Use -1 for unknown categories
    size_grade_encoded = size_grade_mapping.get(Size_Grade,-1)  # Use -1 for unknown categories

    # Prepare input data as a DataFrame for prediction
    input_data = pd.DataFrame([[province_encoded,size_grade_encoded,Weight_Kg,Low_Price,Sales_Total,Stock_On_Hand,month,day]],
                              columns=[Province,Size_Grade,Weight_Kg,Low_Price,Sales_Total,Stock_On_Hand,month,day])
     # Rename columns to string names
     # Make sure the feature names match the model's expectations
    input_data.columns = ['Province', 'Size_Grade', 'Weight_Kg', 'Low_Price', 'Sales_Total', 'Stock_On_Hand', 'month', 'day']

    # Make prediction
    predicted_price = loaded_model.predict(input_data)

    return predicted_price[0]

col1,col2 = st.columns(2)
with col1:
    Province= st.selectbox('Province', ['NORTHERN CAPE', 'WESTERN CAPE - CERES', 'WEST COAST','SOUTH WESTERN FREE STATE', 'WESTERN FREESTATE', 'NATAL',
                                    'KWAZULU NATAL', 'OTHER AREAS', 'TRANSVAAL'])
    Size_Grade= st.selectbox("size grade", ['1M', '2L', '1R', '1L', '1Z', '1S', '1X', '3L', '2R', '2M', '3S','3Z', '3M', '2Z', '3R', '2S'])
    Weight_Kg = st.number_input("weight per kilo", min_value=0.0)
    Low_Price=st.number_input("Low_Price", min_value=0)
with col2:
    Sales_Total= st.number_input('total sale', min_value=0)
    Stock_On_Hand= st.number_input('stock on hand', step=1)
    month = st.slider("Month",1,12)
    day = st.slider("Day",1,31)



# Make prediction
if st.button("Predict"):
     # Call the prediction function
    prediction_price=predict_price(Province,Size_Grade,Weight_Kg,Low_Price,Sales_Total,Stock_On_Hand,month,day)
    st.success(f'Predicted Average Price of ONION BROWN: R{prediction_price:.2f}')
