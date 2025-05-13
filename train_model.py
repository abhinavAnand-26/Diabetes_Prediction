import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Encode categorical columns
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
le = LabelEncoder()
df['smoking_history'] = le.fit_transform(df['smoking_history'])

# Split features and target
X = df.drop(columns='diabetes')
Y = df['diabetes']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
classifier = svm.SVC(kernel='linear', class_weight='balanced')
classifier.fit(X_train, Y_train)

# Compute accuracies
train_acc = accuracy_score(classifier.predict(X_train), Y_train)
test_acc = accuracy_score(classifier.predict(X_test), Y_test)
class_report = classification_report(Y_test, classifier.predict(X_test))

# Save model, scaler, label encoder, and metrics
joblib.dump(classifier, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(train_acc, 'train_accuracy.pkl')
joblib.dump(test_acc, 'test_accuracy.pkl')
joblib.dump(class_report, 'classification_report.pkl')

print("Model and metrics saved successfully.")