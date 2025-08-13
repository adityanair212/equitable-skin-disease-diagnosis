import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = './processed_dataset'
test_dir = os.path.join(data_dir, 'test')
batch_size = 32
confidence_threshold = 0.4  # You can adjust this threshold

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset and loader
test_dataset = ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes

# Load models
def load_model(model_name, path, num_classes):
    if model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model")
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

num_classes = len(class_names)
mobilenet = load_model("mobilenet_v3", "best_mobilenet_model.pth", num_classes)
densenet = load_model("densenet121", "best_densenet121.pth", num_classes)
efficientnet = load_model("efficientnet_b0", "best_efficientnet_model.pth", num_classes)

# Evaluation with hard voting and confidence thresholding
all_preds = []
all_labels = []

print("🔍 Evaluating Ensemble (DenseNet121 + EfficientNetB0 + MobileNetV3) with Hard Voting + Confidence Thresholding...")
for inputs, labels in tqdm(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits_mobilenet = mobilenet(inputs)
        logits_densenet = densenet(inputs)
        logits_efficientnet = efficientnet(inputs)

        probs_mobilenet = torch.softmax(logits_mobilenet, dim=1)
        probs_densenet = torch.softmax(logits_densenet, dim=1)
        probs_efficientnet = torch.softmax(logits_efficientnet, dim=1)

        preds_mobilenet = torch.argmax(probs_mobilenet, dim=1)
        preds_densenet = torch.argmax(probs_densenet, dim=1)
        preds_efficientnet = torch.argmax(probs_efficientnet, dim=1)

        for i in range(inputs.size(0)):
            votes = [preds_mobilenet[i].item(), preds_densenet[i].item(), preds_efficientnet[i].item()]
            vote_counts = np.bincount(votes, minlength=num_classes)
            final_pred = np.argmax(vote_counts)
            avg_conf = (probs_mobilenet[i, final_pred] + probs_densenet[i, final_pred] + probs_efficientnet[i, final_pred]) / 3

            if avg_conf.item() < confidence_threshold:
                all_preds.append(-1)  # Unknown
            else:
                all_preds.append(final_pred)
            all_labels.append(labels[i].item())

# Filter out unknowns for metric computation
filtered_preds = []
filtered_labels = []
unknowns = 0

for pred, label in zip(all_preds, all_labels):
    if pred == -1:
        unknowns += 1
    else:
        filtered_preds.append(pred)
        filtered_labels.append(label)

# Results
print(f"\n❌ Rejected {unknowns}/{len(all_preds)} samples due to low confidence.")
print("\n📋 Classification Report (excluding unknowns):")
print(classification_report(filtered_labels, filtered_preds, target_names=class_names))
print(f"🎯 Accuracy: {accuracy_score(filtered_labels, filtered_preds):.4f}")
print(f"🎯 Precision: {precision_score(filtered_labels, filtered_preds, average='weighted'):.4f}")
print(f"🎯 Recall: {recall_score(filtered_labels, filtered_preds, average='weighted'):.4f}")
print(f"🎯 F1-Score: {f1_score(filtered_labels, filtered_preds, average='weighted'):.4f}")
