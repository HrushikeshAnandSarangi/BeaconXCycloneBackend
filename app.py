from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

import pickle

# Load the model
with open('cyclone.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

if hasattr(model, 'use_label_encoder'):
    model._le = None

# Check if model is loaded
print("Model loaded successfully!")


@app.route('/')
def home():
    return "ML Model API is working!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json

    # Check if 'features' key exists in the data
    if 'features' not in data:
        return jsonify({'error': 'No features provided'}), 400

    # Extract features and convert to numpy array
    features = np.array(data['features']).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    # Return the prediction
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
