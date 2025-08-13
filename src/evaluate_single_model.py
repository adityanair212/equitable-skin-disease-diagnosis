import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from tqdm import tqdm

def evaluate_model(model, model_name, weights_path):
    model.eval()

    test_dir = os.path.join("processed_dataset", "test")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    dataset = ImageFolder(test_dir, transform=transform)
    class_names = dataset.classes
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"[{model_name}] Loaded {len(dataset)} test images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Inference - {model_name}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs / 1.4, dim=1)  # temp scaling
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n📊 [{model_name}] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"✅ Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
    return all_preds, all_labels

# === Evaluate All Models ===
if __name__ == "__main__":
    from torchvision.models import densenet121, mobilenet_v3_large, efficientnet_b0

    # DenseNet121
    model_densenet = densenet121(weights=None)
    model_densenet.classifier = torch.nn.Linear(model_densenet.classifier.in_features, 5)
    evaluate_model(model_densenet, "DenseNet121", "best_densenet121.pth")

    # EfficientNetB0
    model_effnet = efficientnet_b0(weights=None)
    model_effnet.classifier[1] = torch.nn.Linear(model_effnet.classifier[1].in_features, 5)
    evaluate_model(model_effnet, "EfficientNetB0", "best_efficientnet_model.pth")

    # MobileNetV3
    model_mobile = mobilenet_v3_large(weights=None)
    model_mobile.classifier[3] = torch.nn.Linear(model_mobile.classifier[3].in_features, 5)
    evaluate_model(model_mobile, "MobileNetV3", "best_mobilenet_model.pth")
