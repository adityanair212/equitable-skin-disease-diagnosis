# save as: generate_gradcam_all_models_clean.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# Configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dir = "processed_dataset/test"
gradcam_dir = "gradcam_outputs"
os.makedirs(gradcam_dir, exist_ok=True)
image_size = 224
confidence_threshold = 0.4

# Load test dataset
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
dataset = ImageFolder(test_dir, transform=transform)
raw_dataset = ImageFolder(test_dir)  # for original path access
class_names = dataset.classes

loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Model loader
def load_model(name, path):
    model = None
    if name == "densenet":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 5)
    elif name == "efficientnet":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    elif name == "mobilenet":
        model = models.mobilenet_v3_large(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

models_dict = {
    "densenet": load_model("densenet", "best_densenet121.pth"),
    "efficientnet": load_model("efficientnet", "best_efficientnet_model.pth"),
    "mobilenet": load_model("mobilenet", "best_mobilenet_model.pth"),
}

# Grad-CAM processing
for model_name, model in models_dict.items():
    print(f"🔍 Generating Grad-CAMs for {model_name}")
    target_layer = model.features[-1] if "densenet" in model_name else model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    for idx, (input_tensor, label) in enumerate(tqdm(loader)):
        input_tensor = input_tensor.to(device)
        label = label.item()

        # Get filename and class
        orig_path, _ = raw_dataset.samples[idx]
        filename = os.path.basename(orig_path)
        class_folder = class_names[label]

        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
            conf = F.softmax(output, dim=1)
            pred = torch.argmax(conf, dim=1).item()
            confidence = conf[0][pred].item()

        # Only generate for correct + confident predictions
        if pred == label and confidence >= confidence_threshold:
            rgb_img = cv2.imread(orig_path)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (image_size, image_size))
            rgb_img = rgb_img.astype(np.float32) / 255.0

            cam_input = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            grayscale_cam = cam(input_tensor=cam_input.to(device), targets=[ClassifierOutputTarget(label)])
            cam_image = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

            save_dir = os.path.join(gradcam_dir, model_name, class_folder)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, filename), cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
