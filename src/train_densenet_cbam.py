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
from torchvision import datasets, models

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# === CBAM Modules ===
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class DenseNet121_CBAM(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.base = models.densenet121(pretrained=True)
        self.ca = ChannelAttention(1024)
        self.sa = SpatialAttention()
        self.base.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.base.features(x)
        features = self.ca(features) * features
        features = self.sa(features) * features
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.base.classifier(out)

# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha = self.alpha[targets]
            loss = alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# === Config ===
DATA_DIR = "processed_dataset"
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42
MODEL_SAVE_PATH = "best_densenet121_cbam.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# === Albumentations Transforms ===
train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomResizedCrop(
        size=(224, 224),
        scale=(0.7, 1.0),
        ratio=(0.75, 1.33),
        interpolation=cv2.INTER_LINEAR,
        p=1.0
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ColorJitter(p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15), p=0.5),
    A.CoarseDropout(min_holes=1, max_holes=4, max_height=20, max_width=20, fill_value=0, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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

# === Model, Loss, and Optimizer Setup ===
model = DenseNet121_CBAM(num_classes=NUM_CLASSES).to(DEVICE)
for param in model.base.features.parameters():
    param.requires_grad = False  # freeze early layers initially

alpha_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)

# Optimizer now groups the classifier parameters and the attention modules
optimizer = optim.AdamW([
    {'params': model.base.classifier.parameters(), 'lr': LEARNING_RATE},
    {'params': list(model.ca.parameters()) + list(model.sa.parameters()), 'lr': LEARNING_RATE}
])

# === Training Loop with Progressive Unfreezing ===
best_val_acc = 0

for epoch in range(EPOCHS):
    if epoch == 5:
        print("🔓 Unfreezing base layers for fine-tuning")
        for param in model.base.features.parameters():
            param.requires_grad = True

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

    model.eval()
    val_correct = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / len(val_loader.dataset)
    print(f"📊 Epoch {epoch+1} → Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Best model saved.")

print("\n📋 Final Classification Report on Validation Set:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.dataset.classes))
