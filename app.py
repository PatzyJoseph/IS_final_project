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

        # Convert image bytes to a format OpenCV can read (numpy array)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Reads as BGR

        if img_np is None:
            return jsonify({"error": "Could not decode image. Invalid image format?"}), 400

        # --- Perform Prediction using the loaded YOLOv8 model ---
        # The 'predict' method handles preprocessing (resizing, normalization) internally.
        # It returns a list of Results objects (one per image in the batch).
        # We set conf and iou thresholds here.
        results = yolov8_model.predict(
            source=img_np,
            conf=0.30, # Confidence threshold (adjust as needed)
            iou=0.4,   # IoU threshold for Non-Maximum Suppression (adjust as needed)
            classes=None, # Detect all classes (bleeding, redness, swelling)
            verbose=False # Suppress verbose output from ultralytics
        )

        detections_list = []
        annotated_img_b64 = ""

        if results and len(results) > 0:
            result = results[0] # Get the Results object for the first (and only) image

            # --- Get Annotated Image from YOLOv8 ---
            # The .plot() method returns the image with bounding boxes and labels drawn on it.
            # It returns a NumPy array in BGR format.
            annotated_img_np = result.plot()

            # Convert the annotated image from BGR (OpenCV default) to RGB (PIL/frontend expects RGB)
            annotated_img_rgb = cv2.cvtColor(annotated_img_np, cv2.COLOR_BGR2RGB)

            # Convert the annotated image (NumPy array) to a PIL Image, then to bytes, then base64
            pil_img = Image.fromarray(annotated_img_rgb)
            buffered = io.BytesIO()
            pil_img.save(buffered, format='JPEG') # Save as JPEG for web efficiency
            buffered.seek(0)
            annotated_img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # --- Extract Detection Details ---
            # Iterate through detected boxes to get class, confidence, and bbox
            for det in result.boxes:
                # xyxy gives [x1, y1, x2, y2] coordinates
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                confidence = float(det.conf[0])
                class_id = int(det.cls[0])

                symptom_name = SYMPTOM_CLASSES.get(class_id, f"unknown_class_id_{class_id}")

                detections_list.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": symptom_name,
                    "confidence": round(confidence, 2) # Round for cleaner display
                })
            print(f"Detected {len(detections_list)} objects.")
        else:
            print("No objects detected by YOLOv8 model.")

        # Determine overall symptom presence based on detected classes
        classes_detected = [d['class'].lower() for d in detections_list]

        # Define required symptoms
        required_symptoms = ['redness', 'bleeding', 'swelling']

        results_for_frontend = {
            "detections": detections_list,
            "gingivitis": all(cls in classes_detected for cls in required_symptoms),
            "bleeding": 'bleeding' in classes_detected,
            "redness": 'redness' in classes_detected,
            "swelling": 'swelling' in classes_detected # This will now correctly reflect if swelling is detected
        }
        results_for_frontend["image"] = annotated_img_b64 # The base64 encoded image with boxes

        return jsonify(results_for_frontend)

    except Exception as e:
        print(f"Error in predict_route: {e}") # Print error for server-side debugging
        return jsonify({"error": str(e)}), 500

# ---------- Main ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

