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
import pandas as pd
import joblib
from io import StringIO

st.title("User CSV upload")

def handle_csv_prediction():

    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        uploadedData = pd.read_csv(uploaded_file)
        st.write("### First 5 rows of the Uploaded CSV file:")
        st.dataframe(uploadedData.head())

        file_path = r'/workspaces/cropPredictionSys/preTrainedModels/Crops_recommendation.csv'
        df = pd.read_csv(file_path)    


        df.shape
        st.write("======================Column names:===================\n", df.columns)
        st.write("===============Data types:==========================\n", df.dtypes)
        st.write("\n==============Summary statistics=====================:")
        st.write(df.describe())
        st.write("\n=========Missing values:=====================")
        st.write(df.isnull().sum().sum())
        st.write("\n====std=======")
        st.write(df.std(numeric_only=True))
        st.write("\n====unique values=======")
        df.nunique()
        st.write("Average Nitrogen Ratio: {0:.2f}".format(df['N'].mean()))
        st.write("Average Phosphorous Ratio: {0:.2f}".format(df['P'].mean()))
        st.write("Average Potassium Ratio: {0:.2f}".format(df['K'].mean()))
        st.write("Average Temperature (C): {0:.2f}".format(df['temperature'].mean()))
        st.write("Average Humidity: {0:.2f}".format(df['humidity'].mean()))
        st.write("Average pH value: {0:.2f}".format(df['ph'].mean()))
        st.write("Average Rainfall: {0:.2f}".format(df['rainfall'].mean()))
        label_encoder = LabelEncoder()
        df['label_encoded'] = label_encoder.fit_transform(df['label'])

        # Define numeric_columns before using it
        numeric_columns = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label_encoded']]
        df2 = numeric_columns.groupby('label_encoded').mean()
        styled_df = df2.style.background_gradient(cmap='coolwarm')
        styled_df

        # # Correlation Matrix
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(8,8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()
        plt.figure(figsize=(25, 10))
        sns.countplot(x='label', data=df)
        plt.title('Label Distribution', fontsize=20)
        plt.show()

        # # Pie Chart
        # import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        df['label'].value_counts().plot(kind='pie', autopct="%.1f%%")
        plt.title('Label Distribution')
        plt.ylabel('')
        # plt.show()
        st.pyplot(plt)

        # # Histogram
        columns_to_plot = ['N', 'P', 'K', 'temperature', 'humidity', 'ph','rainfall']
        for column in columns_to_plot:
            plt.figure(figsize=(7, 3))
            sns.histplot(df[column], color='brown')
            plt.title(f'Histogram of {column.capitalize()}')
            plt.xlabel(f'{column.capitalize()} Levels')
            plt.ylabel('Frequency')
            # plt.show()
            st.pyplot(plt)

        # # KDE
        plt.figure(figsize=(12,12))
        columns_to_plot = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        i = 1
        for col in columns_to_plot:
            plt.subplot(3,3,i)
            sns.kdeplot(df[col])
            i += 1
        plt.tight_layout()
        # plt.show()
        st.pyplot(plt)

        # # Outliers Detection
        plt.figure(figsize=(12,12))

        columns_to_plot = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        i = 1
        for col in columns_to_plot:

            plt.subplot(3,3,i)
            df[[col]].boxplot()
            i+=1
        mean_ph_by_label = df.groupby('label')['ph'].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=mean_ph_by_label.index, y=mean_ph_by_label.values, palette='winter')
        plt.title('Average pH Required for Each Label (Sorted)')
        plt.xlabel('Label')
        plt.ylabel('Average pH')
        plt.xticks(rotation=45)
        # plt.show()
        st.pyplot(plt)
        top5_max_rainfall_crops = df.groupby('label')['rainfall'].max().nlargest(5)
        plt.figure(figsize=(8, 8))
        plt.pie(top5_max_rainfall_crops, labels=top5_max_rainfall_crops.index, autopct='%1.1f%%', startangle=140)
        plt.title('Proportion of Top 5 Crops with Maximum Rainfall')
        plt.axis('equal')
        # plt.show()
        st.pyplot(plt)
        plt.figure(figsize=(10, 6))
        plt.hist(df['N'], bins=10, alpha=0.5, color='b', label='Nitrogen (N)')
        plt.hist(df['P'], bins=10, alpha=0.5, color='g', label='Phosphorus (P)')
        plt.hist(df['K'], bins=10, alpha=0.5, color='r', label='Potassium (K)')
        plt.xlabel('Nutrient Concentration', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.title('Distribution of NPK Nutrient Concentrations', fontweight='bold')
        plt.legend()
        # plt.show()
        st.pyplot(plt)

        # # Line Graph
        min_max_temp = df.groupby('label').agg({'temperature': ['min', 'max']})
        min_max_temp.columns = ['min_temp', 'max_temp']
        min_max_temp = min_max_temp.reset_index()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=min_max_temp, x='label', y='min_temp', marker='o', label='Min Temperature')

        plt.title('Minimum Temperature Required for Each Crop')
        plt.xlabel('Crop')
        plt.ylabel('Temperature')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        st.pyplot(plt)

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=min_max_temp, x='label', y='max_temp', marker='o', label='Max Temperature', color='orange')

        plt.title('Maximum Temperature Required for Each Crop')
        plt.xlabel('Crop')
        plt.ylabel('Temperature')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        st.pyplot(plt)
        d=df.drop(['label','Growth Characteristics','Use (Food, Feed, Fiber)','Type','Water Requirements','Harvest Method'],axis = 'columns')
        d
        zscores = stats.zscore(d)
        outliers = (zscores > 3).all(axis=1)
        zscore_data = d.drop(d[outliers].index)
        st.write("DataFrame after removing outliers:")
        st.dataframe(zscore_data.head())
        df.columns
        categorical_data = df[['label','Growth Characteristics','Use (Food, Feed, Fiber)','Type','Water Requirements','Harvest Method']]

        df = pd.concat([categorical_data, zscore_data], axis=1)

        st.write("DataFrame with both numerical and categorical columns after removing outliers:")
        st.dataframe(df.head())

        # # One-Hot Encoder
        categorical_cols = ['label']
        numerical_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(df['label'])
        X = df[numerical_cols]
        y = encoded_labels

        # # splitting And Scailing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # # Feature Engineering
        selector = SelectKBest(score_func=f_classif, k=4)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        X_test_selected

        # # Training Models And Accuracy
        # # SVM MODEL
        SVM_Model = joblib.load("/workspaces/cropPredictionSys/preTrainedModels/Model_Training_SVM.pkl")
        y_pred = SVM_Model.predict(X_test_selected)
        svm_accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy: {svm_accuracy * 100:.2f}%')

        # # LOGISTIC REGRESSION
        LR_Model = joblib.load("/workspaces/cropPredictionSys/preTrainedModels/Model_Training_Logistic.pkl")
        y_pred_log = LR_Model.predict(X_test_selected)
        logistic_accuracy = accuracy_score(y_test, y_pred_log)
        st.write(f'Accuracy: {logistic_accuracy * 100:.2f}%')

        # # Decision Tree
        DT_Model = joblib.load("/workspaces/cropPredictionSys/preTrainedModels/Model_Training_Decision.pkl")
        y_pred_dt = DT_Model.predict(X_test_selected)
        dt_accuracy = accuracy_score(y_test, y_pred_dt)
        st.write(f'Accuracy: {dt_accuracy * 100:.2f}%')

        # # K-Nearest Neighbour
        KNN_Model = joblib.load("/workspaces/cropPredictionSys/preTrainedModels/Model_Training_KNN.pkl")
        y_pred_knn = KNN_Model.predict(X_test_selected)
        knn_accuracy = accuracy_score(y_test, y_pred_knn)
        st.write(f'KNN Accuracy: {knn_accuracy * 100:.2f}%')

        # # GussianNB
        Gaussian_Model = joblib.load("/workspaces/cropPredictionSys/preTrainedModels/Model_Training_Gussian.pkl")
        y_pred_nb = Gaussian_Model.predict(X_test_selected)
        nb_accuracy = accuracy_score(y_test, y_pred_nb)
        st.write(f' GussianNB Accuracy: {nb_accuracy * 100:.2f}%')

        # # Random Forest
        RF_Model = joblib.load("/workspaces/cropPredictionSys/preTrainedModels/Model_Training_Random.pkl")
        X_test_selected = selector.transform(X_test_scaled)
        y_pred_rf = RF_Model.predict(X_test_selected)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        st.write(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')

        # # RANDOM FOREST GIVES HIGHEST ACCURACY

        # # Comparison Accuracies Graph
        accuracy_scores = [svm_accuracy, logistic_accuracy, dt_accuracy, knn_accuracy, nb_accuracy, rf_accuracy]

        models = ['SVM', 'Logistic Regression', 'Decision Tree', 'KNN', 'GaussianNB', 'Random Forest']

        plt.figure(figsize=(8, 6))
        plt.title('Accuracy Score', fontweight='bold', fontsize=13, bbox={'facecolor': '0.9', 'pad': 10}, pad=20)
        plt.xlabel('Algorithms', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)

        plt.plot(models, accuracy_scores, color='skyblue', marker="d")
        # plt.show()
        st.pyplot(plt)
        accuracy_scores = [svm_accuracy, logistic_accuracy,rf_accuracy,dt_accuracy,knn_accuracy,nb_accuracy]

        models = ['SVM', 'Logistic Regression','Random Forest','Decision Tree','KNN','GussianNB']
        plt.figure(figsize=(10, 6))
        plt.bar(models, accuracy_scores, color=['blue','red'])
        plt.xlabel('Model')
        plt.ylabel('Accuracy Score')
        plt.title('Comparison of Accuracy Scores between SVM, Logistic Regression,Random Forest,GussianNB')
        plt.ylim(0, 1)

        # # Cross Validation Accuracy
        # # Cross-Validation Accuracy Of SVM
        scores = cross_val_score(SVM_Model, X_train_selected, y_train, cv=5)
        st.write(f'Cross-Validation Accuracy Of SVm: {np.mean(scores)* 100:.2f}%')

        # # Cross-Validation Of Logistic Regression
        scores = cross_val_score(LR_Model, X_train_selected, y_train, cv=5)
        st.write(f'Cross-Validation Accuracy Of Logistic Regression: {np.mean(scores)* 100:.2f}%')

        # # Cross-Validation Accuracy Of Decision Tree
        scores = cross_val_score(DT_Model, X_train_selected, y_train, cv=5)
        st.write(f'Cross-Validation Accuracy Of Decision Tree: {np.mean(scores)* 100:.2f}%')

        # # Cross Validation Accuracy Of KNN
        scores = cross_val_score(KNN_Model, X_train_selected, y_train, cv=5)
        st.write(f'Cross-Validation Accuracy Of KNN: {np.mean(scores)* 100:.2f}%')

        # # Crosss-Validation Accuracy Of GussianNB
        scores = cross_val_score(Gaussian_Model, X_train_selected, y_train, cv=5)
        st.write(f'Cross-Validation Accuracy Of GussianNB: {np.mean(scores)* 100:.2f}%')

        # # Cross-Validation Accuracy Of Random Forest
        scores = cross_val_score(RF_Model, X_train_selected, y_train, cv=5)
        st.write(f'Cross-Validation Accuracy Of Random Forest: {np.mean(scores)* 100:.2f}%')

        # # Cross-Validation Accuracies Graph
        models = ['Random Forest', 'GaussianNB', 'KNN', 'Decision Tree', 'Logistic Regression', 'SVM']
        accuracies = [0.9380681818181819, 0.9329545454545454, 0.9011363636363636, 0.7809699296753891, 0.8920454545454547, 0.9051136363636363]

        plt.figure(figsize=(10, 6))
        plt.bar(models, accuracies, color='blue')
        plt.xlabel('Models')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Cross-Validation Accuracies of Different Models')
        plt.ylim(0.7, 1)
        # plt.show()
        st.pyplot(plt)

        # # Confusion Matrix Of Each Model
        # # CONFUSION MATRIX FOR SVM
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix For SVM')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        # plt.show()
        st.pyplot(plt)

        # # CONFUSION MATRIX FOR LOGISTIC REGRESSION
        cm = confusion_matrix(y_test, y_pred_log)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix For Logistic Regression')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        # plt.show()
        st.pyplot(plt)

        # # CONFUSION MATRIX FOR Decision Tree
        cm = confusion_matrix(y_test, y_pred_dt)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix For Decision Tree')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        # plt.show()
        st.pyplot(plt)

        # # Confusion Matrix For KNN
        cm = confusion_matrix(y_test, y_pred_knn)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix For KNN')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        # plt.show()
        st.pyplot(plt)

        # # Confusion Matrix For GussianNB
        cm = confusion_matrix(y_test, y_pred_nb)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix For GussianNB')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        # plt.show()
        st.pyplot(plt)

        # # Confusion Matrix For Random Forest
        cm = confusion_matrix(y_test, y_pred_rf)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix For Random Forest')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        # plt.show()
        st.pyplot(plt)

        # # Classification Report For Each Model

        # # Classification Report For SVM
        st.write("================Classification Report:===================")
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred)
        st.text("Classification Report for SVM Model:")
        st.text(report)

        # # Classification Report For Logistic Regression
        st.write("=================Classification Report:=====================")
        report = classification_report(y_test, y_pred_log)

        st.text("Classification Report for Logistic Regression Model:")
        st.text(report)

        # # Classification Report For Decision Tree
        st.write("=================Classification Report:=====================")

        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred_dt)

        st.text("Classification Report for Decision Tree Model:")
        st.text(report)

        # # Classification Report For KNN
        st.write("=================Classification Report:=====================")

        from sklearn.metrics import classification_report

        report = classification_report(y_test, y_pred_knn)
        st.text("Classification Report for KNN Model:")
        st.text(report)

        # # Classification Report For GussianNB
        st.write("=================Classification Report:=====================")

        report = classification_report(y_test, y_pred_nb)
        st.text("Classification Report for GussainNB Model:")
        st.text(report)

        # # Classification Report For Random Forest
        st.write("=================Classification Report:=====================")

        report = classification_report(y_test, y_pred_rf)
        st.text("Classification Report for Random Forest Model:")
        st.text(report)

        # Upload the CSV file
        # uploaded = files.upload()

        # Read the uploaded file into a pandas DataFrame
        # file_path = list(uploaded.keys())[0]  # Get the filename
        # data = pd.read_csv(file_path)

        # Display the first few rows of the uploaded data
        # print("Uploaded data:")
        # print(data.head())

        # Ensure that the input data contains the required columns
        required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

        if not all(col in uploadedData.columns for col in required_columns):
            raise ValueError(f"The uploaded file must contain the following columns: {required_columns}")

        # Extract the necessary features from the CSV file
        user_inputs = uploadedData[required_columns]

        # Scale and transform the input data
        user_inputs_scaled = selector.transform(scaler.transform(user_inputs))

        # Initialize lists to store predictions
        svm_preds = []
        logistic_preds = []
        knn_preds = []
        dt_preds = []
        nb_preds = []
        top_rf_crops = []
        growth_characteristics = []
        use_type = []
        type_info = []
        water_requirements = []
        harvest_method = []

        crop_label_data = [
            {'label': 'rice', 'Growth Characteristics': 'Grass', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Cereals',
            'Water Requirements': 'Drought', 'Harvest Method': 'By Hand Or Machine'},
            {'label': 'maize', 'Growth Characteristics': 'Grass', 'Use (Food, Feed, Fiber)': 'Feed, Fiber', 'Type': 'Cereals',
            'Water Requirements': 'Drought', 'Harvest Method': 'By Hand Or Machine'},
            {'label': 'chickpea', 'Growth Characteristics': 'Bush', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Legume',
            'Water Requirements': 'Drought', 'Harvest Method': 'Machine'},
            {'label': 'kidneybeans', 'Growth Characteristics': 'Bush', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Legume',
            'Water Requirements': 'Drought', 'Harvest Method': 'By Hand And Machine'},
            {'label': 'pigeonpeas', 'Growth Characteristics': 'Bush', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Legume',
            'Water Requirements': 'Drought Resistant', 'Harvest Method': 'By Hand'},
            {'label': 'mothbeans', 'Growth Characteristics': 'Bush', 'Use (Food, Feed, Fiber)': 'Fiber', 'Type': 'Legume',
            'Water Requirements': 'Drought Resistant', 'Harvest Method': 'By Hand'},
            {'label': 'mungbean', 'Growth Characteristics': 'Bush', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Legume',
            'Water Requirements': 'Drought', 'Harvest Method': 'By Hand'},
            {'label': 'blackgram', 'Growth Characteristics': 'Bush', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Legume',
            'Water Requirements': 'Drought Resistant', 'Harvest Method': 'By Hand'},
            {'label': 'lentil', 'Growth Characteristics': 'Bush', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Legume',
            'Water Requirements': 'Moderate', 'Harvest Method': 'By Hand'},
            {'label': 'pomegranate', 'Growth Characteristics': 'Tree', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Fruit',
            'Water Requirements': 'Moderate', 'Harvest Method': 'By Hand'},
            {'label': 'banana', 'Growth Characteristics': 'Herbaceous', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Fruit',
            'Water Requirements': 'High', 'Harvest Method': 'By Hand'},
            {'label': 'mango', 'Growth Characteristics': 'Tree', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Fruit',
            'Water Requirements': 'Moderate', 'Harvest Method': 'By Hand'},
            {'label': 'grapes', 'Growth Characteristics': 'Vine', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Fruit',
            'Water Requirements': 'Moderate', 'Harvest Method': 'By Hand'},
            {'label': 'watermelon', 'Growth Characteristics': 'Vine', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Fruit',
            'Water Requirements': 'High', 'Harvest Method': 'By Hand'},
            {'label': 'muskmelon', 'Growth Characteristics': 'Vine', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Fruit',
            'Water Requirements': 'Moderate', 'Harvest Method': 'By Hand'},
            {'label': 'apple', 'Growth Characteristics': 'Tree', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Fruit',
            'Water Requirements': 'Moderate', 'Harvest Method': 'By Hand'},
            {'label': 'orange', 'Growth Characteristics': 'Tree', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Fruit',
            'Water Requirements': 'Moderate', 'Harvest Method': 'By Hand'},
            {'label': 'papaya', 'Growth Characteristics': 'Tree', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Fruit',
            'Water Requirements': 'High', 'Harvest Method': 'By Hand'},
            {'label': 'coconut', 'Growth Characteristics': 'Tree', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Fruit',
            'Water Requirements': 'Moderate', 'Harvest Method': 'By Hand'},
            {'label': 'cotton', 'Growth Characteristics': 'Shrub', 'Use (Food, Feed, Fiber)': 'Fiber', 'Type': 'Commercial',
            'Water Requirements': 'Moderate', 'Harvest Method': 'Machine'},
            {'label': 'jute', 'Growth Characteristics': 'Herbaceous', 'Use (Food, Feed, Fiber)': 'Fiber', 'Type': 'Commercial',
            'Water Requirements': 'High', 'Harvest Method': 'By Hand'},
            {'label': 'coffee', 'Growth Characteristics': 'Shrub', 'Use (Food, Feed, Fiber)': 'Food', 'Type': 'Beverage',
            'Water Requirements': 'Moderate', 'Harvest Method': 'By Hand'}
        ]


        # Convert the data into a pandas DataFrame
        crop_info_df = pd.DataFrame(crop_label_data)

        growth_characteristics_col = 'Growth Characteristics'  # Replace with your actual column name
        use_type_col = 'Use (Food, Feed, Fiber)'  # Replace with your actual column name
        type_info_col = 'Type'  # Replace with your actual column name
        water_requirements_col = 'Water Requirements'  # Replace with your actual column name
        harvest_method_col = 'Harvest Method'  # Replace with your actual column name

        # Rest of the code remains unchanged
        # Make predictions for each row in the file
        for i, user_input_scaled in enumerate(user_inputs_scaled):
            user_input_scaled = user_input_scaled.reshape(1, -1)  # Reshape for prediction

            # Predict using SVM
            user_pred_svm = SVM_Model.predict(user_input_scaled)
            predicted_label_svm = label_encoder.inverse_transform(user_pred_svm)[0]
            svm_preds.append(predicted_label_svm)

            # Predict using Logistic Regression
            user_pred_lr = LR_Model.predict(user_input_scaled)
            predicted_label_lr = label_encoder.inverse_transform(user_pred_lr)[0]
            logistic_preds.append(predicted_label_lr)

            # Predict using KNN
            user_pred_knn = KNN_Model.predict(user_input_scaled)
            predicted_label_knn = label_encoder.inverse_transform(user_pred_knn)[0]
            knn_preds.append(predicted_label_knn)

            # Predict using Decision Tree
            user_pred_dt = DT_Model.predict(user_input_scaled)
            predicted_label_dt = label_encoder.inverse_transform(user_pred_dt.astype(int))[0]
            dt_preds.append(predicted_label_dt)

            # Predict using GaussianNB
            user_pred_nb = Gaussian_Model.predict(user_input_scaled)
            predicted_label_nb = label_encoder.inverse_transform(user_pred_nb)[0]
            nb_preds.append(predicted_label_nb)

            # Predict probabilities using Random Forest
            user_pred_prob_rf = RF_Model.predict_proba(user_input_scaled)
            crop_labels = label_encoder.classes_
            crop_probabilities = dict(zip(crop_labels, user_pred_prob_rf[0]))
            sorted_crop_probabilities = sorted(crop_probabilities.items(), key=lambda x: x[1], reverse=True)

            # Retrieve the top predicted crop
            top_crop = sorted_crop_probabilities[0][0]
            top_rf_crops.append(top_crop)

            # Use if condition to get crop info from Crop Label DataFrame
            crop_info = crop_info_df[crop_info_df['label'].str.strip() == top_crop]
            if not crop_info.empty:
                growth_characteristics.append(crop_info[growth_characteristics_col].values[0])
                use_type.append(crop_info[use_type_col].values[0])
                type_info.append(crop_info[type_info_col].values[0])
                water_requirements.append(crop_info[water_requirements_col].values[0])
                harvest_method.append(crop_info[harvest_method_col].values[0])
            else:
                growth_characteristics.append("Unknown")
                use_type.append("Unknown")
                type_info.append("Unknown")
                water_requirements.append("Unknown")
                harvest_method.append("Unknown")

        # Add predictions and additional details to the DataFrame
        uploadedData['SVM Prediction'] = svm_preds
        uploadedData['Logistic Regression'] = logistic_preds
        uploadedData['KNN Prediction'] = knn_preds
        uploadedData['Decision Tree Prediction'] = dt_preds
        uploadedData['GaussianNB Prediction'] = nb_preds
        uploadedData['Random Forest Top Crop'] = top_rf_crops
        uploadedData['Growth Characteristics'] = growth_characteristics
        uploadedData['Use (Food, Feed, Fiber)'] = use_type
        uploadedData['Type'] = type_info
        uploadedData['Water Requirements'] = water_requirements
        uploadedData['Harvest Method'] = harvest_method

        # Display the updated DataFrame
        st.write("\nPredictions and additional crop information added to the DataFrame:")
        st.write(uploadedData.head())

        # Save the updated DataFrame to a new CSV file
        # uploadedData.to_csv("predicted_crops_with_info.csv", index=False)
        # st.write("\nUpdated predictions and crop details have been saved to 'predicted_crops_with_info.csv'.")

        csv_buffer = StringIO()
        uploadedData.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        # Provide the file for download
        st.download_button(
            label="Download Predicted Crops with Info",
            data=csv_content,
            file_name="predicted_crops_with_info.csv",
            mime="text/csv"
        )

        st.write("\nYou can download the updated predictions and crop details above.")


        # # Mean Squared Error For Each Model
        mse = mean_squared_error(y_test, y_pred_log)
        st.write("Mean Squared Error For Logistic Regression: ", mse)

        mse = mean_squared_error(y_test, y_pred)
        st.write("Mean Squared Error For SVM:", mse)

        mse = mean_squared_error(y_test, y_pred_rf)
        st.write("Mean Squared Error For Random Forest:", mse)

        mse = mean_squared_error(y_test, y_pred_dt)
        st.write("Mean Squared Error For Random Forest:", mse)

        mse = mean_squared_error(y_test, y_pred_knn)
        st.write("Mean Squared Error For Random Forest:", mse)

        mse = mean_squared_error(y_test, y_pred_nb)
        st.write("Mean Squared Error For GussianNB:", mse)

        # # MSE Comparision Graph
        models = ['SVM', 'Logistic Regression','Random Forest','Decison Tree','KNN','GussianNB']
        mse_values = [9.218181818181819, 13.777272727272727,4.377272727272727,6.406818181818182,8.038636363636364,7.593181818181818]

        plt.figure(figsize=(10, 5))
        plt.bar(models, mse_values, color=['blue', 'red'])
        plt.xlabel('Model')
        plt.ylabel('Mean Squared Error')
        plt.title('Mean Squared Error Comparison between SVM,Logistic Regression,Random Forest,Decison Tree,GussianNB')
        # plt.show()
        st.pyplot(plt)

        # # Feature Importance For Each Model
        feature_names = ['N','P','K','temperature', 'humidity', 'ph','rainfall', 'label']
        top_coefficients = LR_Model.coef_[0][-8:]
        top_feature_indices = np.argsort(np.abs(top_coefficients))
        top_feature_names = [feature_names[i] for i in top_feature_indices]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(top_coefficients)), np.abs(top_coefficients), color='blue', align='center')
        plt.xticks(range(len(top_coefficients)), top_feature_names, rotation=45)
        plt.xlabel('Feature')
        plt.ylabel('Absolute Coefficient')
        plt.title('Feature Importance for Logistic Regression')

        svm_coefficients = SVM_Model.coef_[0]
        top_coefficients = svm_coefficients[:8]
        top_feature_indices = np.argsort(np.abs(top_coefficients))
        top_feature_names = [feature_names[i] for i in top_feature_indices]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(top_coefficients)), np.abs(top_coefficients), color='blue', align='center')
        plt.xticks(range(len(top_coefficients)), top_feature_names, rotation=45)
        plt.xlabel('Feature')
        plt.ylabel('Absolute Coefficient')
        plt.title('Feature Importance for SVM')
        # plt.show()
        st.pyplot(plt)

        feature_names = ['N','P','K','temperature', 'humidity', 'ph','rainfall', 'label']
        top_importances = RF_Model.feature_importances_
        top_feature_indices = np.argsort(top_importances)[-8:]
        top_feature_names = [feature_names[i] for i in top_feature_indices]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(top_importances)), top_importances[top_feature_indices], color='blue', align='center')
        plt.xticks(range(len(top_importances)), top_feature_names, rotation=45)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance for Random Forest')
        # plt.show()
        st.pyplot(plt)

        feature_names = ['N','P','K','temperature', 'humidity', 'ph','rainfall', 'label']
        top_importances =DT_Model.feature_importances_
        top_feature_indices = np.argsort(top_importances)[-8:]
        top_feature_names = [feature_names[i] for i in top_feature_indices]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(top_importances)), top_importances[top_feature_indices], color='blue', align='center')
        plt.xticks(range(len(top_importances)), top_feature_names, rotation=45)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance for Decison Tree')
        # plt.show()
        st.pyplot(plt)

        from sklearn.inspection import permutation_importance

        result = permutation_importance(KNN_Model, X_test_selected, y_test, n_repeats=10, random_state=42)

        feature_names = np.array(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        top_feature_indices = result.importances_mean.argsort()[-4:]
        top_feature_names = feature_names[top_feature_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_feature_names)), result.importances_mean[top_feature_indices], color='blue', align='center')
        plt.yticks(range(len(top_feature_names)), top_feature_names)
        plt.xlabel('Mean Decrease in Accuracy')
        plt.title('Permutation Feature Importance for Top 4 Features in KNN')
        # plt.show()
        st.pyplot(plt)

        # # Distribution Graph Of Each Feature By Using GussianNB Model
        means = Gaussian_Model.theta_
        top_feature_indices = np.argsort(np.mean(means, axis=0))[-4:]

        top_feature_names = np.array(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])[top_feature_indices]
        num_classes = means.shape[0]

        fig, axes = plt.subplots(len(top_feature_names), 1, figsize=(6, 4 * len(top_feature_names)))
        for i, feature_idx in enumerate(top_feature_indices):
            feature_name = top_feature_names[i]
            feature_values = means[:, feature_idx]
            axes[i].bar(np.arange(num_classes), feature_values, align='center')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel(f'Mean value of {feature_name}')
            axes[i].set_title(f'Distribution of {feature_name} across classes')
        plt.tight_layout()
        # plt.show()
        st.pyplot(plt)
        unique_labels = df['label'].unique()
        unique_labels


handle_csv_prediction()