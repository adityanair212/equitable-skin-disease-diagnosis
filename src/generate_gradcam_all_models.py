import os
import torch
import numpy as np
import pandas as pd
import cv2
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# === CONFIG ===
IMG_DIR = "processed_dataset/test"
CONFIDENCE_CSV = "confidence_summary.csv"
PRED_CSV = "ensemble_predictions.csv"
MODEL_PATHS = {
    "densenet": "best_densenet121.pth",
    "efficientnet": "best_efficientnet_model.pth",
    "mobilenet": "best_mobilenet_model.pth"
}
OUTPUT_DIR = "gradcam_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load and prepare data ===
class_names = sorted([d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))])
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

# Load both CSVs and merge them
df_conf = pd.read_csv(CONFIDENCE_CSV)
df_preds = pd.read_csv(PRED_CSV)
df_preds["true_label_name"] = df_preds["true_label"].map(idx_to_class)
df_preds["predicted_name"] = df_preds["predicted_label"].map(idx_to_class)

df = df_conf.copy()
df["image"] = df_preds["image"]
df["true_label"] = df_preds["true_label_name"]
df["predicted"] = df_preds["predicted_name"]

# === Preprocessing ===
def load_image_for_gradcam(image_path):
    img = cv2.imread(image_path)[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    return img, preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# === Load model based on type ===
def load_model(model_type):
    if model_type == "densenet":
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(class_names))
        target_layer = model.features[-1]
    elif model_type == "mobilenet":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(class_names))
        target_layer = model.features[-1]
    elif model_type == "efficientnet":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
        target_layer = model.features[-1]
    else:
        raise ValueError("Unknown model type")

    model.load_state_dict(torch.load(MODEL_PATHS[model_type], map_location=device))
    model.to(device).eval()
    return model, target_layer

# === Generate Grad-CAM and save ===
def generate_and_save_gradcam(model, target_layer, img_path, class_idx, out_path):
    raw_img, input_tensor = load_image_for_gradcam(img_path)
    input_tensor = input_tensor.to(device)
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])
    cam_image = show_cam_on_image(raw_img, grayscale_cam[0], use_rgb=True)
    cv2.imwrite(out_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

# === Main loop ===
for model_name in MODEL_PATHS:
    print(f"\n🔍 Generating Grad-CAMs for: {model_name}")
    model, target_layer = load_model(model_name)
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    for cls in class_names:
        cls_dir = os.path.join(model_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        correct = df[(df["true_label"] == cls) & (df["predicted"] == cls)].sort_values("confidence", ascending=False).head(2)
        incorrect = df[(df["true_label"] == cls) & (df["predicted"] != cls)].head(2)
        samples = pd.concat([correct, incorrect])

        for i, row in samples.iterrows():
            filename = row.get("image")
            if not filename:
                continue
            img_path = os.path.join(IMG_DIR, cls, filename)
            if not os.path.exists(img_path):
                continue
            pred_idx = class_to_idx.get(row["predicted"], class_to_idx[cls])
            out_path = os.path.join(cls_dir, f"{i}_{row['predicted']}.png")
            try:
                generate_and_save_gradcam(model, target_layer, img_path, pred_idx, out_path)
            except Exception as e:
                print(f"⚠️ Failed on {filename}: {e}")

print("\n✅ All Grad-CAMs saved in:", OUTPUT_DIR)
