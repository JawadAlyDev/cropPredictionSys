import streamlit as st

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
import time

st.set_page_config(
    page_title="Home - Crop Predict",
    page_icon="🌾"
)

st.title("Home")
st.sidebar.success("Select a page above to navigate")


# file_path = r'/workspaces/cropPredictionSys/Crops_recommendation.csv'
# df = pd.read_csv(file_path)
# st.write("First 5 rows of the dataset:")
# st.dataframe(df.head())

st.write("## <<< Welcome to Crop Prediction System >>>")

st.write("### Choose an option:")

option = st.selectbox("Select Action", ["Select", "Upload CSV for Prediction", "Provide Manual Input for Prediction"])

if option == 'Upload CSV for Prediction':
    st.write("You selected: Upload CSV for Prediction")
    time.sleep(3)
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        df = pd.read_csv(uploaded_file)
        st.write("### First 5 rows of the Uploaded CSV file:")
        st.dataframe(df.head())
    
        
elif option == 'Provide Manual Input for Prediction':
    st.write("You selected: Provide Manual Input for Prediction")
    time.sleep(3)

    if "nitrogenQty" not in st.session_state and "phosphorousQty" not in st.session_state and "potassiumQty" not in st.session_state and "temperatureQty" not in st.session_state and "humidityQty" not in st.session_state and "pHQty" not in st.session_state and "rainfallQty" not in st.session_state:
        st.session_state["nitrogenQty"] = ""
        st.session_state["phosphorousQty"] = ""
        st.session_state["potassiumQty"] = ""
        st.session_state["temperatureQty"] = ""
        st.session_state["humidityQty"] = ""
        st.session_state["pHQty"] = ""
        st.session_state["rainfallQty"] = ""

    nitrogen = st.text_input('Enter Nitrogen Quantity for Crop:')
    phosphorous = st.text_input('Enter Phosphorous Quantity for Crop:')
    potassium = st.text_input('Enter Potassium Quantity for Crop:')
    temperature = st.text_input('Enter Temperature Quantity for Crop:')
    humidity = st.text_input('Enter Humidity Quantity for Crop:')
    pH = st.text_input('Enter pH Quantity for Crop:')
    rainfall = st.text_input('Enter Rainfall Quantity for Crop:')

    submitBtn = st.button("Submit")
    if submitBtn:
        st.session_state["nitrogenQty"] = float(nitrogen)
        st.session_state["phosphorousQty"] = float(phosphorous)
        st.session_state["potassiumQty"] = float(potassium)
        st.session_state["temperatureQty"] = float(temperature)
        st.session_state["humidityQty"] = float(humidity)
        st.session_state["pHQty"] = float(pH)
        st.session_state["rainfallQty"] = float(rainfall)
        st.write("You have entered ",nitrogen,phosphorous,potassium,temperature,humidity,pH,rainfall)
        

