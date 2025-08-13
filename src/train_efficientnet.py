import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_dir = './processed_dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
save_path = 'best_efficientnet_model.pth'

# Image size and hyperparams
image_size = 224
batch_size = 32
num_epochs = 20
learning_rate = 1e-4

# Augmentations
train_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ColorJitter(p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15), p=0.5),
    A.CoarseDropout(max_holes=4, max_height=20, max_width=20, min_holes=1, fill_value=0, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Custom Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir)
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.dataset.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, label

    def __len__(self):
        return len(self.dataset)

# Datasets
train_dataset = CustomDataset(train_dir, transform=train_transform)
val_dataset = CustomDataset(val_dir, transform=val_transform)

# Compute weights for class balancing
targets = [label for _, label in train_dataset]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
sample_weights = [class_weights[label] for label in targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model
model = models.efficientnet_b0(pretrained=True)
num_classes = len(train_dataset.dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / len(val_loader.dataset)

    print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print("✅ Best model saved.")

# Final Report
print("\n📋 Final Classification Report on Validation Set:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.dataset.classes))
