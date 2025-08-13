import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURATION ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dir = "./processed_dataset/test"
batch_size = 32
num_classes = 5
confidence_threshold = 0.4

# Model-specific temperature scaling
temperature_scaling = {
    "mobilenet": 1.3,
    "densenet": 1.4,
    "efficientnet": 1.2
}

# Ensemble weights (based on validation/test performance)
ensemble_weights = {
    "mobilenet": 0.25,
    "densenet": 0.4,
    "efficientnet": 0.35
}

# Enable hard voting fallback
use_hard_voting = False  # ← Set True to toggle

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- LOAD DATA ----------------
test_dataset = ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)

# ---------------- LOAD MODELS ----------------
def load_model(model_name, path):
    if model_name == "mobilenet":
        model = models.mobilenet_v3_large(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "densenet":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Invalid model name")

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

models_dict = {
    "mobilenet": load_model("mobilenet", "best_mobilenet_model.pth"),
    "densenet": load_model("densenet", "best_densenet121.pth"),
    "efficientnet": load_model("efficientnet", "best_efficientnet_model.pth")
}

# ---------------- ENSEMBLE EVALUATION ----------------
all_preds = []
all_labels = []
all_confidences = []

print("🧠 Evaluating Ensemble with", "Hard Voting" if use_hard_voting else "Soft Voting + Temperature Scaling")
for inputs, labels in tqdm(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        softmax_probs = {}
        predictions = {}

        for name, model in models_dict.items():
            logits = model(inputs)
            temp = temperature_scaling[name]
            probs = torch.softmax(logits / temp, dim=1)
            softmax_probs[name] = probs
            predictions[name] = torch.argmax(probs, dim=1)

        if not use_hard_voting:
            ensemble_probs = torch.zeros_like(probs).to(device)
            for name in softmax_probs:
                ensemble_probs += ensemble_weights[name] * softmax_probs[name]

            max_probs, final_preds = torch.max(ensemble_probs, dim=1)
        else:
            final_preds = []
            max_probs = []
            for i in range(inputs.size(0)):
                votes = [predictions[name][i].item() for name in predictions]
                vote_counts = np.bincount(votes, minlength=num_classes)
                final = np.argmax(vote_counts)
                conf = sum(softmax_probs[name][i][final].item() for name in softmax_probs) / len(models_dict)
                final_preds.append(final)
                max_probs.append(conf)
            final_preds = torch.tensor(final_preds).to(device)
            max_probs = torch.tensor(max_probs).to(device)

        for pred, prob, label in zip(final_preds, max_probs, labels):
            all_labels.append(label.item())
            all_confidences.append(prob.item())
            all_preds.append(pred.item() if prob.item() >= confidence_threshold else -1)

# ---------------- METRICS ----------------
filtered_preds = []
filtered_labels = []
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

# ---------------- EXPORT RESULTS FOR VISUALIZATION ----------------
confidence_df = pd.DataFrame({
    "true_label": [class_names[i] for i in all_labels],
    "predicted": [class_names[i] if i != -1 else "unknown" for i in all_preds],
    "confidence": all_confidences
})
confidence_df.to_csv("confidence_summary.csv", index=False)

predictions_df = pd.DataFrame({
    "image": [f"{i}.png" for i in range(len(all_preds))],
    "true_label": all_labels,
    "predicted_label": all_preds
})
predictions_df.to_csv("ensemble_predictions.csv", index=False)

print("\n✅ Exported: confidence_summary.csv and ensemble_predictions.csv")
