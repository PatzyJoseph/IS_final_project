from flask import Flask, render_template, request, jsonify
from flask import send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
import io
import os
import base64

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

    print("Output shape:", output_data.shape)
    print("Sample values:", output_data[0, :5])  # show 5 detections
    
    return postprocess(output_data)


def draw_boxes(image_bytes, detections):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Your model was trained on 640x640, make sure the image matches that
    image = image.resize((640, 640))
    width, height = image.size

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        box = [x1, y1, x2, y2]
        label = det['class']
        conf = det['confidence']
        
        # Draw rectangle
        draw.rectangle(box, outline="red", width=2)

        # Draw label
        text = f"{label}: {conf:.2f}"
        draw.text((x1, y1 - 10), text, fill="red")

    return image

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
    try:
        file = request.files['file']
        image_bytes = file.read()
        detections = predict(image_bytes)

        parsed_detections = [{
            "bbox": [float(x) for x in det['bbox']],
            "class": det['class'],
            "confidence": float(det['confidence'])
        } for det in detections]

        classes_detected = [d['class'].lower() for d in parsed_detections]

        results = {
            "detections": parsed_detections,
            "gingivitis": any(cls in classes_detected for cls in ['redness', 'swelling', 'bleeding']),
            "bleeding": 'bleeding' in classes_detected,
            "redness": 'redness' in classes_detected,
            "swelling": 'swelling' in classes_detected
        }

        # Draw image and convert to base64
        image_with_boxes = draw_boxes(image_bytes, parsed_detections)
        buf = io.BytesIO()
        image_with_boxes.save(buf, format='JPEG')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        results["image"] = image_base64

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Main ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
