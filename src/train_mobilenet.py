import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, datasets
from torchvision.models import MobileNet_V3_Large_Weights
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from tqdm import tqdm

def main():
    # === Config ===
    DATA_DIR = "processed_dataset"
    BATCH_SIZE = 32
    NUM_CLASSES = 5
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH = "best_mobilenet_model.pth"
    SEED = 42

    # === Set seeds ===
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # === Albumentations Transforms ===
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-15, 15), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ])

    class AlbumentationsDataset(torch.utils.data.Dataset):
        def __init__(self, folder_path, transform):
            self.dataset = datasets.ImageFolder(folder_path)
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img_path, label = self.dataset.imgs[idx]
            image = datasets.folder.default_loader(img_path)
            image = np.array(image)
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, label

    # === Datasets ===
    train_dataset = AlbumentationsDataset(os.path.join(DATA_DIR, "train"), train_transform)
    val_dataset = AlbumentationsDataset(os.path.join(DATA_DIR, "val"), val_transform)

    # === Class-balanced weighting ===
    targets = [label for _, label in train_dataset]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # === Weighted Sampler ===
    class_counts = np.bincount(targets)
    weights = 1. / class_counts
    sample_weights = [weights[label] for label in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # === DataLoaders ===
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # === Model ===
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # === Loss + Optimizer ===
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # === Training ===
    best_val_acc = 0.0
    print("\n🔁 Starting training...\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"🟢 Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # === Validation ===
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"🔵 Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / total
        print(f"\n📊 Epoch {epoch+1} Summary → Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # === Save best model ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Best model saved.\n")

    # === Final Classification Report ===
    print("\n📋 Final Classification Report on Validation Set:")
    print(classification_report(all_labels, all_preds, target_names=[
        "basal-cell-carcinoma", "eczema", "melanocytic-nevi", "seborrheic-keratosis", "verruca-vulgaris"
    ]))

# ✅ Fix multiprocessing issue
if __name__ == "__main__":
    main()
