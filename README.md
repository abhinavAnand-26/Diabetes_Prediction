# Diabetes_Prediction

# **Overview**
The primary goal of this project is to predict the likelihood of an individual having diabetes based on various health-related attributes. Utilizing machine learning techniques, specifically the Support Vector Machine (SVM) algorithm, the model analyzes input data to provide accurate predictions.

# **Project Structure**

The repository comprises the following key files:

app.py: Flask web application to interact with the trained model.

train_model.py: Script to train the SVM model using the dataset.

diabetes_prediction_2.0.ipynb: Jupyter Notebook detailing the exploratory data analysis and model training process.

diabetes_prediction_dataset.csv: Dataset containing health-related attributes for model training.

svm_model.pkl: Serialized SVM model for prediction.

scaler.pkl: Serialized scaler object used for feature scaling.

label_encoder.pkl: Serialized label encoder for categorical data.

classification_report.pkl: Serialized classification report of the model's performance.

train_accuracy.pkl & test_accuracy.pkl: Serialized training and testing accuracy scores.

requirements.txt: List of dependencies required to run the project.

img.jpeg: Image used in the web application's interface.

# **Model Performance**
The SVM model was evaluated using standard classification metrics. The serialized classification_report.pkl contains detailed performance metrics, while train_accuracy.pkl and test_accuracy.pkl provide the training and testing accuracy scores, respectively
