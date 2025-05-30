from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------- Config ----------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE           = (640, 640)          # adapt to what your model expects
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.tflite")

# ---------- App ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB upload limit

# ---------- Load TFLite model once, when the server starts ----------
interpreter = tf.lite.Interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- Helpers ----------
def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((640, 640))  # YOLOv8 input size
    image_np = np.array(image, dtype=np.float32) / 255.0  # normalize
    image_np = np.expand_dims(image_np, axis=0)  # (1, 640, 640, 3)
    return image_np

import numpy as np

SYMPTOM_CLASSES = {0: "bleeding", 1: "redness", 2: "swelling"}
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess(output_data):
    detections = []
    output = np.squeeze(output_data)  # (8400, 7)
    for i in range(output.shape[0]):
        x, y, w, h, objectness, *class_scores = output[i]
        class_scores = sigmoid(np.array(class_scores))
        objectness = sigmoid(objectness)
        score = objectness * np.max(class_scores)
        class_idx = np.argmax(class_scores)
        if score >= CONFIDENCE_THRESHOLD:
            cx, cy = x, y
            x1 = max(0, cx - w / 2)
            y1 = max(0, cy - h / 2)
            x2 = cx + w / 2
            y2 = cy + h / 2
            detections.append({
                "class": SYMPTOM_CLASSES.get(class_idx, str(class_idx)),
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
    return detections
    
def predict(image_bytes):
    input_data = preprocess_image(image_bytes)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # shape (1, 8400, 7)

    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("Output shape:", output_data.shape)
    print("Sample values:", output_data[0, :5])  # show 5 detections
    
    return postprocess(output_data)

# ---------- Routes ----------
@app.route("/")
def index():
    # Flask automatically serves templates/index.html
    return render_template("index.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed(file.filename):
        return jsonify({"error": "No selected image / wrong format"}), 400

    try:
        result = predict(file.read())
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Inference failed")
        return jsonify({"error": str(e)}), 500

# ---------- Main ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
