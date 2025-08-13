import os
import uuid
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# --- Directories ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
GRADCAM_FOLDER = os.path.join('static', 'gradcams')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# --- Labels ---
LABELS = [
    'Basal Cell Carcinoma',
    'Eczema',
    'Melanocytic Nevi',
    'Seborrheic Keratosis',
    'Verruca Vulgaris'
]

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- Model Loader ---
def load_model(path, model_type):
    if model_type == 'densenet':
        m = models.densenet121(pretrained=False)
        m.classifier = torch.nn.Linear(m.classifier.in_features, 5)
    elif model_type == 'mobilenet':
        m = models.mobilenet_v3_large(pretrained=False)
        m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, 5)
    elif model_type == 'efficientnet':
        m = models.efficientnet_b0(pretrained=False)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 5)
    else:
        raise ValueError("Unknown model type")
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m

# --- Load Ensemble Models Once ---
densenet_model  = load_model("best_densenet121.pth", "densenet")
mobilenet_model = load_model("best_mobilenet_model.pth", "mobilenet")
effnet_model    = load_model("best_efficientnet_model.pth", "efficientnet")

# --- Grad-CAM Utilities ---
def generate_gradcam(model, img_tensor, target_class):
    gradients = []
    activations = []

    def forward_hook(module, inp, outp):
        activations.append(outp)
        outp.register_hook(lambda grad: gradients.append(grad))

    # Hook the feature extractor
    handle = model.features.register_forward_hook(forward_hook)

    model.zero_grad()
    output = model(img_tensor)
    score = output[0, target_class]
    score.backward()
    handle.remove()

    grad = gradients[0][0].cpu().numpy()
    act = activations[0][0].cpu().detach().numpy()
    weights = np.mean(grad, axis=(1,2))
    cam = np.sum(weights[:,None,None] * act, axis=0)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def overlay_gradcam(image_path, cam, save_path):
    orig = cv2.imread(image_path)
    orig = cv2.resize(orig, (224,224))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (224,224))
    superimposed = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, superimposed)

# --- Prediction Logic ---
def predict(image_path, threshold=0.4):
    img = Image.open(image_path).convert('RGB')
    inp = transform(img).unsqueeze(0)

    with torch.no_grad():
        preds = [F.softmax(m(inp), dim=1).cpu().numpy()[0]
                 for m in (densenet_model, mobilenet_model, effnet_model)]
    preds = np.array(preds)
    votes = np.argmax(preds, axis=1)
    counts = np.bincount(votes, minlength=5)
    majority = np.argmax(counts)
    confidence = counts[majority] / len(votes)
    avg_conf = preds.mean(axis=0)

    # Generate Grad-CAM from DenseNet
    cam = generate_gradcam(densenet_model, inp, majority)
    gc_name = f"{uuid.uuid4().hex}.jpg"
    gc_path = os.path.join(GRADCAM_FOLDER, gc_name)
    overlay_gradcam(image_path, cam, gc_path)

    return {
        "prediction": LABELS[majority] if confidence >= threshold else "UNCERTAIN",
        "confidence": float(confidence),
        "all_labels": LABELS,
        "all_confidences": avg_conf.tolist(),
        "gradcam_url": f"/static/gradcams/{gc_name}"
    }

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_predict():
    if 'file' not in request.files:
        return jsonify({"error":"No file part"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error":"No selected file"}), 400

    # Save upload
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # Predict + Grad-CAM
    result = predict(save_path)
    result["image_url"] = f"/static/uploads/{filename}"
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
