import os
import numpy as np
from PIL import Image
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# Initialize App
app = Flask(__name__)
CORS(app) 

# Load Model and Labels
try:
    interpreter = tf.lite.Interpreter(model_path="assets/skin_disease30.tflite")
    interpreter.allocate_tensors()
    runner = interpreter.get_signature_runner()
    input_details = runner.get_input_details()
    input_key = list(input_details.keys())[0]
    shape = input_details[input_key]['shape']
    height, width = shape[1], shape[2]
    
    with open("assets/labels21.txt", 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")

@app.route("/")
def health_check():
    return "API is running!"

@app.route("/predict", methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((width, height))
        
        input_data = np.expand_dims(img, axis=0)
        if input_details[input_key]['dtype'] == tf.float32:
            input_data = input_data.astype(np.float32)

        prediction_result = runner.run({input_key: input_data})
        output_key = list(prediction_result.keys())[0]
        probabilities = prediction_result[output_key][0]
        
        confidences = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
        sorted_results = sorted(confidences.items(), key=lambda item: item[1], reverse=True)
        top_prediction = sorted_results[0]
        
        return jsonify({ "condition": top_prediction[0], "confidence": top_prediction[1] })
    except Exception as e:
        return jsonify({'error': 'Failed to process image', 'details': str(e)}), 500

# ...