from flask import Flask, render_template, request, jsonify
from flask import send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont # ImageFont imported but not used for custom fonts here
import io
import os
import base64

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------- Config ----------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE           = (640, 640)        # Model expects this input size (width, height)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.tflite")

# ---------- App ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB upload limit

# ---------- Load TFLite model once, when the server starts ----------
try:
    interpreter = tf.lite.Interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"TFLite model loaded successfully from: {MODEL_PATH}")
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    # Exit or handle error appropriately if model fails to load
    exit(1) # Or raise an exception if you prefer Flask to handle it

# ---------- Helpers ----------
def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    """
    Preprocesses the input image: opens, converts to RGB, resizes to IMG_SIZE,
    normalizes pixel values to [0, 1], and adds a batch dimension.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)  # Resize to model input size (640, 640)
    image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension (1, 640, 640, 3)
    return image_np

# --- IMPORTANT: TEMPORARY ADJUSTMENT FOR 2 CLASSES ---
# Your TFLite model's output indicates it only provides 2 class scores
# (7 values per detection: 4 bbox + 1 objectness + 2 class scores).
# This means your 'swelling' class might not be detected or correctly mapped.
# The permanent fix requires re-exporting your YOLOv8 model to TFLite
# ensuring it outputs all 3 classes (which would typically result in 8 values per detection).
# For now, we adjust SYMPTOM_CLASSES to silence the critical warning and attempt detection
# with the classes the model IS outputting.
# Assuming class 0 is 'bleeding' and class 1 is 'redness' based on your previous definition.
SYMPTOM_CLASSES = {0: "bleeding", 1: "redness"}
# -----------------------------------------------------

# --- IMPORTANT: ADJUST THESE THRESHOLDS ---
# Higher CONFIDENCE_THRESHOLD means fewer, but more certain, detections.
# Lower IOU_THRESHOLD means NMS is more aggressive in removing overlapping boxes.
CONFIDENCE_THRESHOLD = 0.30 # Lowered from 0.40 to try and get some detections
IOU_THRESHOLD = 0.4         # Keeping this at 0.4 for now
# ------------------------------------------

def sigmoid(x):
    """Applies the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def non_max_suppression(detections, iou_threshold):
    """
    Applies Non-Maximum Suppression (NMS) to filter out redundant bounding boxes.
    Args:
        detections (list): A list of dictionaries, each containing 'bbox', 'confidence', and 'class'.
        iou_threshold (float): The Intersection Over Union (IoU) threshold for suppression.
    Returns:
        list: A list of filtered detections after NMS.
    """
    if not detections:
        return []

    # Sort detections by confidence score (descending)
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    boxes = np.array([d['bbox'] for d in detections])
    scores = np.array([d['confidence'] for d in detections])
    
    # Map class names back to numerical IDs for NMS logic
    # This assumes SYMPTOM_CLASSES values are unique
    class_name_to_id = {v: k for k, v in SYMPTOM_CLASSES.items()}
    class_ids = np.array([class_name_to_id[d['class']] for d in detections])

    # Convert boxes to (x1, y1, x2, y2) format
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    
    final_indices = []

    # Apply NMS per class to ensure objects of different classes don't suppress each other
    unique_classes = np.unique(class_ids)
    for cls_id in unique_classes:
        # Get indices of detections belonging to the current class
        class_specific_indices = np.where(class_ids == cls_id)[0]
        
        # Sort scores for current class in descending order
        class_scores_sorted_indices = class_specific_indices[np.argsort(scores[class_specific_indices])[::-1]]

        while len(class_scores_sorted_indices) > 0:
            # Select the detection with the highest score
            best_idx = class_scores_sorted_indices[0]
            final_indices.append(best_idx)

            # Calculate IoU with remaining detections in the same class
            current_x1 = x1[best_idx]
            current_y1 = y1[best_idx]
            current_x2 = x2[best_idx]
            current_y2 = y2[best_idx]
            current_area = areas[best_idx]

            # Intersection coordinates
            ix1 = np.maximum(current_x1, x1[class_scores_sorted_indices[1:]])
            iy1 = np.maximum(current_y1, y1[class_scores_sorted_indices[1:]])
            ix2 = np.minimum(current_x2, x2[class_scores_sorted_indices[1:]])
            iy2 = np.minimum(current_y2, y2[class_scores_sorted_indices[1:]])

            intersection_areas = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            
            # Union of the two boxes
            union_areas = current_area + areas[class_scores_sorted_indices[1:]] - intersection_areas

            # Calculate IoU
            iou = intersection_areas / (union_areas + 1e-6) # Add epsilon to avoid division by zero

            # Remove detections with high IoU
            to_remove = np.where(iou > iou_threshold)[0] + 1 # +1 because we sliced sorted_indices[1:]
            
            class_scores_sorted_indices = np.delete(class_scores_sorted_indices, np.concatenate(([0], to_remove)))
            
    # Return only the detections that were kept by NMS
    return [detections[i] for i in sorted(final_indices)] # Sort for consistent output order


def postprocess(output_data):
    """
    Post-processes the raw model output to extract bounding box detections.
    This includes:
    1. Squeezing and transposing the output.
    2. Applying sigmoid to objectness and class scores.
    3. Filtering by confidence threshold.
    4. Converting YOLO-style (normalized center_x, center_y, width, height)
       to (pixel x1, pixel y1, pixel x2, pixel y2).
    5. Applying Non-Maximum Suppression (NMS).
    """
    initial_detections = []
    
    # Common YOLOv8 TFLite output shapes: (1, 8400, 7) or (1, 7, 8400)
    # Adjust output processing based on the actual output_details[0]['shape']
    output_shape = output_details[0]['shape']
    
    # Determine the number of class scores based on the actual model output shape
    # This logic is crucial for aligning with the model's actual output.
    # It assumes the last part of the detection vector (after 4 bbox + 1 objectness) are class scores.
    num_class_scores_from_model = output_shape[-1] - 5
    
    # Attempt to transpose based on whether the 'detections' dimension is 2nd or 3rd
    if output_shape[1] == (4 + 1 + num_class_scores_from_model): # (1, N_detections, 4+1+num_classes)
        output = np.squeeze(output_data)  # Remove batch dim: (N_detections, 4+1+num_classes)
        # No transpose needed
    elif output_shape[2] == (4 + 1 + num_class_scores_from_model): # (1, 4+1+num_classes, N_detections)
        output = np.squeeze(output_data).T # Remove batch dim, then transpose: (N_detections, 4+1+num_classes)
    else:
        # Fallback if shape is truly unexpected, might still lead to issues
        print(f"WARNING: Unexpected output shape from model: {output_shape}. Assuming transpose and proceeding.")
        output = np.squeeze(output_data).T 

    print(f"Processed output shape: {output.shape}. Model reports {num_class_scores_from_model} class scores.")
    # The CRITICAL WARNING below is now handled by setting SYMPTOM_CLASSES to match num_class_scores_from_model
    # but the comment about the model issue remains for clarity.
    if num_class_scores_from_model != len(SYMPTOM_CLASSES):
        print(f"WARNING: The number of class scores from the model ({num_class_scores_from_model}) "
              f"does not match the current SYMPTOM_CLASSES definition ({len(SYMPTOM_CLASSES)}). "
              f"This means you might not be detecting all intended classes or mapping them incorrectly. "
              f"Please ensure your TFLite model outputs the correct number of classes.")


    for i in range(output.shape[0]):
        # The first 4 values are bounding box (cx, cy, w, h) - assumed normalized [0,1]
        # The 5th value is objectness score
        # The remaining values are class scores
        x_norm, y_norm, w_norm, h_norm, objectness, *class_scores = output[i]
        
        # Ensure class_scores list matches the expected length from SYMPTOM_CLASSES
        # This will prevent errors if 'swelling' was expected but not present in model output
        if len(class_scores) != len(SYMPTOM_CLASSES):
            # This case ideally shouldn't happen if SYMPTOM_CLASSES is adjusted to num_class_scores_from_model
            # but it's a safeguard.
            # If the model gives 2 scores but SYMPTOM_CLASSES has 3, this would cause issues.
            # With the fix above (SYMPTOM_CLASSES matching model output), this specific issue is less likely.
            print(f"DEBUG: Mismatch in class_scores length {len(class_scores)} vs SYMPTOM_CLASSES length {len(SYMPTOM_CLASSES)} for detection {i}. Skipping.")
            continue


        # Apply sigmoid to scores (they are usually raw logits)
        class_scores = sigmoid(np.array(class_scores))
        objectness = sigmoid(objectness)

        # Calculate final confidence score
        score = objectness * np.max(class_scores) # Combined confidence
        class_idx = np.argmax(class_scores)       # Get the class with the highest score

        if score >= CONFIDENCE_THRESHOLD: # Filter by confidence
            # Convert normalized YOLO output to pixel coordinates (scale to IMG_SIZE)
            # This is the most common scaling for YOLOv8 TFLite outputs
            cx = x_norm * IMG_SIZE[0]
            cy = y_norm * IMG_SIZE[1]
            w = w_norm * IMG_SIZE[0]
            h = h_norm * IMG_SIZE[1]

            # Convert center (cx, cy), width (w), height (h) to (x1, y1, x2, y2)
            x1 = max(0, cx - w / 2)
            y1 = max(0, cy - h / 2)
            x2 = min(IMG_SIZE[0], cx + w / 2)
            y2 = min(IMG_SIZE[1], cy + h / 2)

            # Ensure valid bounding box (non-zero area)
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                initial_detections.append({
                    "class": SYMPTOM_CLASSES.get(class_idx, f"unknown_idx_{class_idx}"), # Use f-string for unknown
                    "confidence": float(score),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

                # Debugging: Print initial detections before NMS
                # print(f"Raw Detected: Class={SYMPTOM_CLASSES.get(class_idx, f'unknown_idx_{class_idx}')}, "
                #       f"Conf={score:.2f}, Box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    
    print(f"Number of initial detections (before NMS): {len(initial_detections)}")
    
    # Apply Non-Maximum Suppression
    final_detections = non_max_suppression(initial_detections, IOU_THRESHOLD)
    
    return final_detections


def predict(image_bytes):
    """
    Performs inference on the input image bytes using the TFLite model.
    """
    input_data = preprocess_image(image_bytes)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("\n--- Raw Model Output Inspection ---")
    print("Raw Output shape:", output_data.shape)
    # Print a small slice to inspect raw values for debugging bounding box scaling
    # Ensure it's not empty before trying to slice
    if output_data.size > 0:
        # Determine the number of class scores based on the actual model output shape
        num_class_scores_from_model_check = output_details[0]['shape'][-1] - 5
        
        # Check if the model output is (1, N, M) or (1, M, N)
        if output_data.shape[1] == (4 + 1 + num_class_scores_from_model_check):
            print("Sample raw output (first 5 rows/detections):")
            print(output_data[0, :min(5, output_data.shape[1]), :])
        elif output_data.shape[2] == (4 + 1 + num_class_scores_from_model_check):
            print("Sample raw output (first 5 columns/detections after transpose consideration):")
            print(output_data[0, :, :min(5, output_data.shape[2])])
        else:
            print("Output shape is unusual, printing first 5 elements directly:")
            print(output_data.flatten()[:min(35, output_data.size)]) # Print 5 detections flattened

    print("-----------------------------------\n")

    detections = postprocess(output_data)
    
    print(f"\n--- Final Detections after Postprocessing & NMS ({len(detections)}) ---")
    if not detections:
        print("No detections found after postprocessing and NMS.")
    for det in detections:
        print(f"Final Detection: Class={det['class']}, Conf={det['confidence']:.2f}, Bbox={det['bbox']}")
        # Check if coordinates are within reasonable range for visualization
        x1, y1, x2, y2 = det['bbox']
        if not (0 <= x1 <= IMG_SIZE[0] and 0 <= y1 <= IMG_SIZE[1] and \
                0 <= x2 <= IMG_SIZE[0] and 0 <= y2 <= IMG_SIZE[1]):
            print(f"WARNING: Final bbox {det['bbox']} is outside {IMG_SIZE} range!")
        if x1 >= x2 or y1 >= y2:
            print(f"WARNING: Final bbox {det['bbox']} has zero or negative width/height!")
    print("----------------------------------------------------------\n")
    
    return detections


def draw_boxes(image_bytes, detections):
    """
    Draws bounding boxes and labels on the image.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Resize the image to the model's input size before drawing
    # This ensures the bounding box coordinates (which are 640x640 based)
    # align correctly with the image content.
    image = image.resize(IMG_SIZE)
    
    draw = ImageDraw.Draw(image)

    # You can load a custom font here if needed, e.g.:
    # try:
    #     font = ImageFont.truetype("arial.ttf", 15) # Replace with your font path
    # except IOError:
    #     font = None # Fallback to default font (PIL's default font is often small)

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        box = [x1, y1, x2, y2]
        label = det['class']
        conf = det['confidence']
        
        # Draw rectangle
        draw.rectangle(box, outline="red", width=2) # Reverted to drawing all detections

        # Draw label text
        text = f"{label}: {conf:.2f}"
        # Use font=font if you loaded a custom font, otherwise default will be used
        draw.text((x1, y1 - 10), text, fill="red") 

    # --- ADDED FOR LOCAL DEBUGGING ---
    # Saves the image with drawn boxes to the Flask app's directory.
    # This helps confirm if drawing is happening correctly before sending to client.
    try:
        debug_output_path = "debug_output.jpg"
        image.save(debug_output_path)
        print(f"Debug image saved to: {debug_output_path}")
    except Exception as e:
        print(f"Error saving debug image: {e}")
    # ----------------------------------

    return image


# ---------- Routes ----------
@app.route("/")
def index():
    # Flask automatically serves templates/index.html
    return render_template("analysis.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        file = request.files['file']
        image_bytes = file.read()
        
        # Perform prediction and get detections
        detections = predict(image_bytes)

        # Prepare detections for JSON response
        parsed_detections = [{
            "bbox": [float(x) for x in det['bbox']],
            "class": det['class'],
            "confidence": float(det['confidence'])
        } for det in detections]

        # Determine overall symptom presence based on detected classes
        # This will now only check for 'bleeding' and 'redness'
        classes_detected = [d['class'].lower() for d in parsed_detections]

        results = {
            "detections": parsed_detections,
            "gingivitis": any(cls in classes_detected for cls in ['redness', 'bleeding']), # 'swelling' removed for now
            "bleeding": 'bleeding' in classes_detected,
            "redness": 'redness' in classes_detected,
            "swelling": 'swelling' in classes_detected # This will always be False if 'swelling' isn't detected by the model
        }

        # Draw bounding boxes on the original image and encode to base64
        image_with_boxes = draw_boxes(image_bytes, parsed_detections)
        buf = io.BytesIO()
        image_with_boxes.save(buf, format='JPEG') # Save as JPEG for web
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        results["image"] = image_base64

        return jsonify(results)

    except Exception as e:
        print(f"Error in predict_route: {e}") # Print error for server-side debugging
        return jsonify({"error": str(e)}), 500

# ---------- Main ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)