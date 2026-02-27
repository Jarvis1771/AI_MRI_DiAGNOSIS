from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

from pdf_report import generate_medical_report

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "models/mri_model.pth"
OUTPUT_PDF = "static/AI_Medical_Report.pdf"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# MODEL LOADING
# -----------------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

class_names = ["no", "yes"]  # fixed: was ["Abnormal Brain", "Normal Brain"] which was reversed

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_mri(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    label = class_names[predicted_class.item()]
    confidence_percent = float(confidence.item() * 100)

    return label, confidence_percent

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        prediction, confidence = predict_mri(filepath)

        # Generate PDF
        generate_medical_report(
            image_path=filepath,
            prediction=prediction,
            confidence=confidence,
            heatmap_path=None,
            output_pdf=OUTPUT_PDF
        )

        return render_template(
            "result.html",
            image_path=filepath,
            prediction=prediction,
            confidence=confidence,
            pdf_path=OUTPUT_PDF
        )

    return render_template("index.html")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
