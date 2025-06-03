from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import io
import base64
import numpy as np
import cv2 # For image processing (reading, converting color spaces)
from PIL import Image # For image handling with base64 encoding
import pickle # For loading the .pkl model
from ultralytics import YOLO # For the YOLOv8 model object and its predict method

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # Ignore TensorFlow deprecation warnings if any remain

# ---------- Config ----------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
# IMG_SIZE is now primarily for reference, YOLOv8's predict handles resizing internally
# but it's good to know the expected input size for the model.
IMG_SIZE = (640, 640) # Ensure this matches the imgsz used during model export

# Path to your PKL model file
# Make sure 'my_yolov8_model.pkl' is in a 'model' subdirectory relative to app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "my_yolov8_model.pkl")

# ---------- App ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 # 5 MB upload limit

# ---------- Load the YOLOv8 model (from .pkl file) once, when the server starts ----------
yolov8_model = None # Initialize model as None
try:
    with open(MODEL_PATH, 'rb') as f:
        yolov8_model = pickle.load(f)
    print(f"YOLOv8 model loaded successfully from: {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}. Please ensure your 'my_yolov8_model.pkl' is in the 'model' directory.")
    exit(1) # Exit if model is not found, as the app cannot function
except Exception as e:
    print(f"ERROR: An error occurred while loading the YOLOv8 model from PKL: {e}")
    print("Please ensure 'ultralytics' and 'torch' are installed in your Flask environment.")
    exit(1) # Exit if model loading fails for other reasons

# ---------- Symptom Classes ----------
# This MUST match the classes your YOLOv8 model was trained on and exported with.
# Based on your data.yaml, it's 3 classes.
SYMPTOM_CLASSES = {
    0: "bleeding",
    1: "redness",
    2: "swelling"
}

# ---------- Routes ----------
@app.route("/")
def index():
    # Flask automatically serves templates/index.html
    return render_template("index.html")

@app.route('/assets/<path:filename>')
def custom_static(filename):
    return send_from_directory('assets', filename)

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if yolov8_model is None:
        return jsonify({"error": "Model not loaded. Server error."}), 500

    try:
        file = request.files['file']
        image_bytes = file.read()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            return jsonify({"error": "Could not decode image. Invalid image format?"}), 400

        results = yolov8_model.predict(
            source=img_np,
            conf=0.30,
            iou=0.4,
            classes=None,
            verbose=False
        )

        detections_list = []
        annotated_img_b64 = ""

        if results and len(results) > 0:
            result = results[0]
            annotated_img_np = result.plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img_np, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(annotated_img_rgb)
            buffered = io.BytesIO()
            pil_img.save(buffered, format='JPEG')
            buffered.seek(0)
            annotated_img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            for det in result.boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                confidence = float(det.conf[0])
                class_id = int(det.cls[0])

                symptom_name = SYMPTOM_CLASSES.get(class_id, f"unknown_class_id_{class_id}")

                detections_list.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": symptom_name,
                    "confidence": round(confidence, 2)
                })

            print(f"Detected {len(detections_list)} objects.")
        else:
            print("No objects detected by YOLOv8 model.")

        classes_detected = [d['class'].lower() for d in detections_list]

        # Individual symptoms
        bleeding = 'bleeding' in classes_detected
        redness = 'redness' in classes_detected
        swelling = 'swelling' in classes_detected

        # Build combination keys
        symptom_combinations = []
        if bleeding:
            symptom_combinations.append("bleeding")
        if redness:
            symptom_combinations.append("redness")
        if swelling:
            symptom_combinations.append("swelling")
        if bleeding and redness:
            symptom_combinations.append("bleeding_redness")
        if redness and swelling:
            symptom_combinations.append("redness_swelling")
        if bleeding and swelling:
            symptom_combinations.append("bleeding_swelling")
        if bleeding and redness and swelling:
            symptom_combinations.append("gingivitis")

        results_for_frontend = {
            "detections": detections_list,
            "image": annotated_img_b64,
            "symptom_combinations": symptom_combinations,
            "bleeding": bleeding,
            "redness": redness,
            "swelling": swelling,
            "gingivitis": bleeding and redness and swelling
        }

        return jsonify(results_for_frontend)

    except Exception as e:
        print(f"Error in predict_route: {e}")
        return jsonify({"error": str(e)}), 500


# ---------- Main ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

