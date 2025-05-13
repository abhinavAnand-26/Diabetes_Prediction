import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
from PIL import Image
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load saved model, scaler, and label encoder
try:
    classifier = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Error: Model files ('svm_model.pkl', 'scaler.pkl', 'label_encoder.pkl') not found. Please ensure they are in the project directory.")
    st.stop()

# Load dataset for summary statistics and correlation heatmap (optional)
try:
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    # Encode categorical columns for consistency
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
    df['smoking_history'] = le.transform(df['smoking_history'])
    mean_by_outcome = df.groupby('diabetes').mean()
    df_corr = df.select_dtypes(include=['number']).corr()  # Compute correlation matrix
except FileNotFoundError:
    st.warning("Warning: 'diabetes_prediction_dataset.csv' not found. Dataset summary and heatmap will not be displayed.")
    df = None
    mean_by_outcome = None
    df_corr = None

# Streamlit App
def app():
    # Load and display image
    try:
        img = Image.open("img.jpeg")
        img = img.resize((300, 200))
        st.image(img, caption="Diabetes Image", width=300)
    except FileNotFoundError:
        st.warning("Warning: 'img.jpeg' not found. Skipping image display.")

    st.title('Diabetes Prediction')

    st.sidebar.title('Input Features')
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
    age = st.sidebar.slider('Age', 0, 100, 30)
    hypertension = st.sidebar.selectbox('Hypertension', [0, 1])
    heart_disease = st.sidebar.selectbox('Heart Disease', [0, 1])
    smoking_history = st.sidebar.selectbox('Smoking History', list(le.classes_))
    bmi = st.sidebar.slider('BMI', 10.0, 70.0, 25.0)
    hba1c = st.sidebar.slider('HbA1c Level', 3.0, 10.0, 5.5)
    glucose = st.sidebar.slider('Blood Glucose Level', 50, 300, 100)

    # Input validation
    if bmi < 10 or bmi > 70:
        st.error("BMI must be between 10 and 70.")
        st.stop()
    if hba1c < 3 or hba1c > 10:
        st.error("HbA1c Level must be between 3 and 10.")
        st.stop()
    if glucose < 50 or glucose > 300:
        st.error("Blood Glucose Level must be between 50 and 300.")
        st.stop()

    # Encode inputs
    gender_val = {'Male': 1, 'Female': 0, 'Other': 2}[gender]
    smoking_val = le.transform([smoking_history])[0]

    # Create input DataFrame with column names to avoid UserWarning
    input_data = pd.DataFrame(
        [[gender_val, age, hypertension, heart_disease, smoking_val, bmi, hba1c, glucose]],
        columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    )
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = classifier.predict(input_scaled)

    st.write('### Prediction Result:')
    if prediction[0] == 1:
        st.warning('This person **has** diabetes.')
    else:
        st.success('This person **does not** have diabetes.')

    # Display model performance
    st.header('Model Performance')
    st.write("Note: Accuracy and classification report are based on the test set during model training.")
    try:
        train_acc = joblib.load('train_accuracy.pkl')
        test_acc = joblib.load('test_accuracy.pkl')
        class_report = joblib.load('classification_report.pkl')
        st.write(f'Train accuracy: {train_acc:.2f}')
        st.write(f'Test accuracy: {test_acc:.2f}')
        st.write('Classification Report:')
        st.text(class_report)
    except FileNotFoundError:
        st.warning("Warning: Model performance metrics ('train_accuracy.pkl', 'test_accuracy.pkl', 'classification_report.pkl') not found.")
        if df is not None:  # Fallback: Compute metrics on-the-fly if dataset is available
            st.write("Attempting to compute performance metrics on-the-fly...")
            X = df.drop(columns='diabetes')
            Y = df['diabetes']
            X_scaled = scaler.transform(X)
            X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)
            train_acc = accuracy_score(classifier.predict(X_train), Y_train)
            test_acc = accuracy_score(classifier.predict(X_test), Y_test)
            class_report = classification_report(Y_test, classifier.predict(X_test))
            st.write(f'Train accuracy: {train_acc:.2f}')
            st.write(f'Test accuracy: {test_acc:.2f}')
            st.write('Classification Report:')
            st.text(class_report)
        else:
            st.write("Cannot compute metrics without the dataset. Please generate and include the metric files.")

    # Display dataset summary and heatmap (if available)
    if df is not None:
        st.header('Dataset Summary')
        st.write(df.describe())

        st.header('Mean Values by Outcome')
        st.write(mean_by_outcome)

        st.header('Feature Correlation Heatmap')
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Feature Correlation Heatmap")
        st.pyplot(plt)

if __name__ == '__main__':
    app()
