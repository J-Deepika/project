from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
import cv2 

app = Flask(__name__)

# Load pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image = Image.open(file.stream)
    image = image.resize((224, 224))
    image_array = np.expand_dims(np.array(image), axis=0)
    preprocessed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    predictions = model.predict(preprocessed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
    return jsonify(decoded_predictions[0])

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    if not file:
        return jsonify({"error": "Empty file"}), 400
    try:
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.expand_dims(np.array(image), axis=0)
        preprocessed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        predictions = model.predict(preprocessed_image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
        return jsonify(decoded_predictions[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def init_db():
    conn = sqlite3.connect('images.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS images
                      (id INTEGER PRIMARY KEY, image_name TEXT, description TEXT)''')
    conn.close()

init_db()

@app.route('/upload', methods=['POST'])
def upload_image():
    # ... previous code ...
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3) # type: ignore
    
    # Save metadata to the database
    conn = sqlite3.connect('images.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO images (image_name, description) VALUES (?, ?)",
                   (files.filename, str(decoded_predictions[0]))) # type: ignore
    conn.commit()
    conn.close()
    
    return jsonify(decoded_predictions[0])

def process_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Further processing...
    return gray_image