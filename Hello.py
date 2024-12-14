import streamlit as st

# st.title("🎈 My new app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

# homePage config
st.set_page_config(
    page_title="Home - Crop Predict",
    page_icon="🌾"
)

st.title("Home")
st.sidebar.success("Select a page above")

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

file_path = r'/workspaces/cropPredictionSys/Crops_recommendation.csv'
df = pd.read_csv(file_path)
st.write("First 5 rows of the dataset:")
st.dataframe(df.head())

