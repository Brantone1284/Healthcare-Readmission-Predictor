# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, jsonify
import openai
import os

# Set OpenAI API key (replace with your actual key)
openai.api_key = os.getenv('OPENAI_API_KEY')  # Or hardcode: 'your-openai-api-key-here'

# Step 1: Load and Preprocess the Data
def load_and_preprocess_data(file_path='hospital_readmissions.csv'):
    # Load the CSV data
    df = pd.read_csv("hospital_readmissions.csv")
    
    # Handle missing values in A1C_Result (replace nan with 'Unknown')
    df['A1C_Result'] = df['A1C_Result'].fillna('Unknown')
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Admission_Type', 'Diagnosis', 'A1C_Result']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Drop Patient_ID as it's not useful for prediction
    df = df.drop('Patient_ID', axis=1)
    
    # Encode target variable
    df['Readmitted'] = df['Readmitted'].map({'Yes': 1, 'No': 0})
    
    # Features and target
    X = df.drop('Readmitted', axis=1).values
    y = df['Readmitted'].values
    
    # Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler, label_encoders

# Step 2: Build an Advanced TensorFlow Model (Neural Network with Dropout and Early Stopping)
def build_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        Dropout(0.3),  # Dropout for regularization
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Step 3: Train the Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(X_train.shape[1])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on test set
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model.save('readmission_model.h5')
    
    return model, X_test, y_test

# Step 4: Use OpenAI for Advanced Features (e.g., Generate Explanation for Predictions)
def generate_explanation(prediction, patient_data):
    prompt = f"""
    The model predicted that the patient will {'be readmitted' if prediction == 1 else 'not be readmitted'}.
    Patient details: {patient_data}
    Provide a detailed, human-readable explanation for this prediction, including potential risk factors based on age, diagnosis, etc.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Or use 'gpt-3.5-turbo' for cost-efficiency
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message['content']

# Step 5: Flask App for Deployment
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('readmission_model.h5')

# Assuming scaler and label_encoders are saved or reloaded (for simplicity, assume they are available)
# In production, save them with joblib or pickle

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expect JSON input with patient features
    
    # Preprocess input (encode categoricals and scale)
    # Note: In production, use the same label_encoders and scaler
    # For demo, assume input is already preprocessed or add preprocessing logic here
    
    features = np.array([data['features']])  # Example: expects 'features' as list of values
    prediction_prob = model.predict(features)[0][0]
    prediction = 1 if prediction_prob > 0.5 else 0
    
    # Generate explanation using OpenAI
    explanation = generate_explanation(prediction, data)
    
    return jsonify({
        'prediction': 'Yes' if prediction == 1 else 'No',
        'probability': float(prediction_prob),
        'explanation': explanation
    })

if __name__ == '__main__':
    # First, train the model if not already trained
    if not os.path.exists('readmission_model.h5'):
        X, y, scaler, label_encoders = load_and_preprocess_data()
        train_model(X, y)
    
    # Run Flask app
    app.run(debug=True)