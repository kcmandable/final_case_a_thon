import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Lazy-load model (only when needed)
model = None

def get_model():
    global model
    if model is None:
        # Use smallest YOLO model to stay under Render’s memory limits
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
    image = Image.open(image_file.stream)

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
    port = int(os.environ.get("PORT", 5000))  # Required for Render
    app.run(host="0.0.0.0", port=port)

