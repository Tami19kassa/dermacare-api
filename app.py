import os
import numpy as np
from PIL import Image
import io
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# --- Initialize Flask App and Enable CORS ---
app = Flask(__name__)
# This is crucial to allow your React app to call the API
CORS(app) 

# --- Load Model and Labels ---
try:
    model_path = os.path.join(os.path.dirname(__file__), "assets/skin_disease30.tflite")
    labels_path = os.path.join(os.path.dirname(__file__), "assets/labels21.txt")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    height = input_details['shape'][1]
    width = input_details['shape'][2]
    print("Model and labels loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")
    # In a real app, you'd want more robust error handling here
    
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
        if input_details['dtype'] == np.float32:
            input_data = (np.float32(input_data) / 255.0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probabilities = output_data[0]
        
        confidences = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
        
        sorted_results = sorted(confidences.items(), key=lambda item: item[1], reverse=True)
        top_prediction = sorted_results[0]
        
        return jsonify({
            "condition": top_prediction[0],
            "confidence": top_prediction[1]
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))