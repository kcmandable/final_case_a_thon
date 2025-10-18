import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ allow frontend (GitHub Pages or Render) access
from ultralytics import YOLO
from PIL import Image
from pillow_heif import register_heif_opener  # ✅ add HEIC/HEIF image support

# Enable HEIC file reading via Pillow
register_heif_opener()

app = Flask(__name__)
CORS(app)  # ✅ enable CORS globally

# Lazy-load YOLO model
model = None

def get_model():
    global model
    if model is None:
        # Use smallest YOLO model to stay within Render memory limits
        model = YOLO("yolov8n.pt")
    return model


@app.route("/")
def home():
    return jsonify({"message": "YOLO model API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    try:
        image = Image.open(image_file.stream)
    except Exception as e:
        return jsonify({"error": f"Failed to open image: {str(e)}"}), 400

    model = get_model()
    results = model.predict(image)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": int(box.cls),
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()
        })

    return jsonify({"detections": detections})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
