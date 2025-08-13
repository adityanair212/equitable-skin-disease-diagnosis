import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2

# === Config ===
DATA_DIR = "processed_dataset"
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42
MODEL_SAVE_PATH = "best_densenet121.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Reproducibility ===
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Albumentations Transforms ===
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ColorJitter(p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15), p=0.5),
    A.CoarseDropout(min_holes=1, max_holes=4, max_height=20, max_width=20, fill_value=0, p=0.5),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# === Custom Dataset ===
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.dataset.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        return image, label

    def __len__(self):
        return len(self.dataset)

# === Load Datasets ===
train_dataset = AlbumentationsDataset(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = AlbumentationsDataset(os.path.join(DATA_DIR, "val"), transform=val_transform)

# === Compute Class Weights and Sampler ===
targets = [label for _, label in train_dataset]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
sample_weights = [class_weights[label] for label in targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# === Loaders ===
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Model ===
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model.to(DEVICE)

# === Loss and Optimizer ===
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# === Training ===
best_val_acc = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_correct = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_loader.dataset)

    # === Validation ===
    model.eval()
    val_correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / len(val_loader.dataset)
    print(f"\n📊 Epoch {epoch+1} Summary → Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Best model saved.")

# === Final Classification Report ===
print("\n📋 Final Classification Report on Validation Set:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.dataset.classes))
