import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Configure CORS
from flask_cors import CORS

CORS(app, resources={r"/*": {"origins": ["http://localhost", "http://localhost:3000", "http://localhost:63342"]}})#(add "http://localhost:63342" since frontend is there and server is at 8000)

# Load the TensorFlow SavedModel
MODEL = tf.saved_model.load("../model/14", tags="serve")

CLASS_NAMES = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"]


@app.route("/")
def index():
    # Main page
    return render_template("ben.html")


@app.route("/ping")
def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file:
        image = read_file_as_image(file.read())

        # Resize the image to (224, 224)
        image = tf.image.resize(image, [224, 224])

        img_batch = np.expand_dims(image, axis=0)

        # Get the prediction signature
        batch_prediction = MODEL(img_batch)
        predicted_label = np.argmax(batch_prediction)
        confidence = np.max(batch_prediction[0])
        confidence_float=float(confidence)
        confidence_round = round(confidence_float, 4)
        confidence_final=100*confidence_round
        print(batch_prediction)
        return jsonify({
            'class': CLASS_NAMES[predicted_label],
            'confidence': confidence_final
        })


if __name__ == "__main__":
    app.run(host='localhost', port=5000,debug=True)
