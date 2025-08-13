import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = './processed_dataset'
test_dir = os.path.join(data_dir, 'test')
batch_size = 32
num_classes = 5
confidence_threshold = 0.4
temperature = 1.4

# Weights for the ensemble (mobilenet, densenet, efficientnet)
ensemble_weights = torch.tensor([0.3, 0.4, 0.3], device=device)
ensemble_weights = ensemble_weights / ensemble_weights.sum()

# Output directory
results_dir = "results_visuals"
os.makedirs(results_dir, exist_ok=True)

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- DATASET ----------------
test_dataset = ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes
label_map = {i: name for i, name in enumerate(class_names)}
label_map[-1] = "unknown"

# ---------------- LOAD MODELS ----------------
def load_model(name, path):
    if name == "mobilenet":
        model = models.mobilenet_v3_large(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "densenet":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Invalid model name")

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

mobilenet = load_model("mobilenet", "best_mobilenet_model.pth")
densenet = load_model("densenet", "best_densenet121.pth")
efficientnet = load_model("efficientnet", "best_efficientnet_model.pth")

# ---------------- ENSEMBLE EVALUATION ----------------
all_preds, all_labels, all_confidences = [], [], []
softmax_eff, softmax_dense, softmax_mobile = [], [], []

print("🔍 Evaluating Ensemble with Soft Voting + Temperature Scaling...")
for inputs, labels in tqdm(test_loader):
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        logits_m = mobilenet(inputs)
        logits_d = densenet(inputs)
        logits_e = efficientnet(inputs)

        probs_m = F.softmax(logits_m / temperature, dim=1)
        probs_d = F.softmax(logits_d / temperature, dim=1)
        probs_e = F.softmax(logits_e / temperature, dim=1)

        softmax_mobile.append(probs_m.cpu().numpy())
        softmax_dense.append(probs_d.cpu().numpy())
        softmax_eff.append(probs_e.cpu().numpy())

        ensemble_probs = (ensemble_weights[0] * probs_m +
                          ensemble_weights[1] * probs_d +
                          ensemble_weights[2] * probs_e)

        for i in range(inputs.size(0)):
            pred = torch.argmax(ensemble_probs[i]).item()
            conf = ensemble_probs[i, pred].item()
            label = labels[i].item()

            all_labels.append(label)
            all_confidences.append(conf)

            if conf < confidence_threshold:
                all_preds.append(-1)
            else:
                all_preds.append(pred)

# ---------------- METRICS ----------------
filtered_preds, filtered_labels = [], []
unknowns = 0

for p, l in zip(all_preds, all_labels):
    if p == -1:
        unknowns += 1
    else:
        filtered_preds.append(p)
        filtered_labels.append(l)

print(f"\n❌ Rejected {unknowns}/{len(all_preds)} samples due to low confidence.")
print("\n📋 Classification Report (excluding unknowns):")
print(classification_report(filtered_labels, filtered_preds, target_names=class_names))
print(f"🎯 Accuracy: {accuracy_score(filtered_labels, filtered_preds):.4f}")
print(f"🎯 Precision: {precision_score(filtered_labels, filtered_preds, average='weighted'):.4f}")
print(f"🎯 Recall: {recall_score(filtered_labels, filtered_preds, average='weighted'):.4f}")
print(f"🎯 F1-Score: {f1_score(filtered_labels, filtered_preds, average='weighted'):.4f}")

# ---------------- EXPORT FILES FOR VISUALIZATION ----------------
# Save confidence_summary.csv
df_conf = pd.DataFrame({
    "true_label": [label_map[l] for l in all_labels],
    "predicted": [label_map[p] for p in all_preds],
    "confidence": all_confidences
})
df_conf.to_csv(os.path.join(results_dir, "confidence_summary.csv"), index=False)
print("✅ Saved: confidence_summary.csv")

# Save ensemble_predictions.csv
df_preds = pd.DataFrame({
    "image": [f"{i}.png" for i in range(len(filtered_preds))],
    "true_label": filtered_labels,
    "predicted_label": filtered_preds
})
df_preds.to_csv(os.path.join(results_dir, "ensemble_predictions.csv"), index=False)
print("✅ Saved: ensemble_predictions.csv")

# Save softmax outputs
np.save(os.path.join(results_dir, "effnet_softmax.npy"), np.vstack(softmax_eff))
np.save(os.path.join(results_dir, "densenet_softmax.npy"), np.vstack(softmax_dense))
np.save(os.path.join(results_dir, "mobilenet_softmax.npy"), np.vstack(softmax_mobile))
print("✅ Saved: softmax outputs for ROC curves")

# ========= ADD THIS TO BOTTOM OF evaluate_ensemble_final.py =========

from torchvision import transforms
from PIL import Image

def predict_ensemble(pil_img, model_paths, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Takes a PIL image and runs ensemble prediction with TTA.

    Returns:
        softmax_probs (np.array): array of size (5,)
        predicted_class_idx (int)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(pil_img).unsqueeze(0).to(device)

    models_to_load = {
        "efficientnet": models.efficientnet_b0,
        "mobilenetv3": models.mobilenet_v3_large,
        "densenet121": models.densenet121
    }

    weights = {
        "efficientnet": 0.35,
        "mobilenetv3": 0.35,
        "densenet121": 0.30
    }

    softmax = torch.nn.Softmax(dim=1)
    ensemble_probs = torch.zeros((1, 5)).to(device)

    for key in model_paths:
        model_fn = models_to_load[key]
        model = model_fn(weights=None)
        if key == "mobilenetv3":
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 5)
        elif key == "efficientnet":
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)
        else:  # densenet121
            model.classifier = torch.nn.Linear(model.classifier.in_features, 5)

        model.load_state_dict(torch.load(model_paths[key], map_location=device))
        model.to(device)
        model.eval()

        with torch.no_grad():
            logits = model(image_tensor)
            probs = softmax(logits)
            ensemble_probs += weights[key] * probs

    ensemble_probs = ensemble_probs.cpu().numpy()[0]
    pred_class = int(np.argmax(ensemble_probs))
    return ensemble_probs, pred_class
