import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)

# URLs to download models and classes
MODEL_URL = "https://huggingface.co/kcmandable/convnext_tiny_benthic/resolve/main/convnext_tiny_best.pth"
SECOND_MODEL_URL = "https://huggingface.co/kcmandable/benthic_yolov8/resolve/main/benthic_yolov8_best.pt"
CLASSES_URL = "https://huggingface.co/kcmandable/convnext_tiny_benthic/resolve/main/classes.json"

# Global variables for models
yolo_model = None
classifier_model = None
class_names = []

# Ensure directory for models
os.makedirs("benthic_artifacts", exist_ok=True)

def download_file(url, dest):
    """Download a file if it doesn't exist."""
    if not os.path.exists(dest):
        print(f"🔽 Downloading: {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(dest, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved to {dest}")
    return dest

def load_models():
    global yolo_model, classifier_model, class_names

    # --- YOLO ---
    yolo_path = download_file(SECOND_MODEL_URL, "benthic_artifacts/benthic_yolov8_best.pt")
    yolo_model = YOLO(yolo_path)

    # --- ConvNeXt ---
    classifier_path = download_file(MODEL_URL, "benthic_artifacts/convnext_tiny_best.pth")
    model = models.convnext_tiny(pretrained=False)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(num_ftrs, 5)  # adjust if needed
    model.load_state_dict(torch.load(classifier_path, map_location="cpu"))
    model.eval()
    classifier_model = model

    # --- Classes ---
    classes_path = download_file(CLASSES_URL, "benthic_artifacts/classes.json")
    with open(classes_path, "r") as f:
        class_names = json.load(f)

    print("✅ Models and class labels loaded successfully!")


# Image transformation for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route("/")
def home():
    return jsonify({"message": "Benthic detection + classification API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file.stream).convert("RGB")

    if yolo_model is None or classifier_model is None:
        load_models()

    results = yolo_model.predict(image)
    detections = []

    os.makedirs("static", exist_ok=True)
    output_path = "static/output.jpg"
    results[0].save(filename=output_path)

    # Classify YOLO crops
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image.crop((x1, y1, x2, y2))
        input_tensor = transform(cropped).unsqueeze(0)

        with torch.no_grad():
            outputs = classifier_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_class = torch.max(probs, 1)

        label = class_names[pred_class] if pred_class < len(class_names) else str(pred_class.item())
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "yolo_confidence": float(box.conf),
            "classifier_label": label,
            "classifier_confidence": float(conf)
        })

    response = {
        "detections": detections,
        "image_url": request.host_url + "static/output.jpg"
    }
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
