from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Initialize the Flask app
app = Flask(__name__)

# Path to YOLO model
model_path = r"C:\Users\dell\Desktop\render-demo\best (3).pt"  
model = YOLO(model_path)

# Create a directory for saving results
output_dir = "static/results"
os.makedirs(output_dir, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Check if a file is uploaded
    if 'image' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('home'))

    # Read the uploaded image
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Perform detection using YOLO
    results = model.predict(image)
    detected_image = results[0].plot()

    # Save the detected image
    detected_image_path = os.path.join(output_dir, "detected_image.jpg")
    cv2.imwrite(detected_image_path, detected_image)

    # Return the detection result to the HTML page
    return render_template(
        'index.html',
        detected_image_url=detected_image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
