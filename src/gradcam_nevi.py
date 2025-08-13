import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def generate_gradcam(model, image_pil, class_idx, target_layer_name='features.denseblock4'):
    """
    Generate GradCAM heatmap for a given PIL image and model.
    """

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Hook to get feature maps and gradients
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        output.register_hook(save_gradient)
        activations.append(output)

    # Register hook
    for name, module in model.named_modules():
        if name == target_layer_name:
            module.register_forward_hook(forward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    # Compute Grad-CAM
    grad = gradients[0].cpu().data.numpy()[0]
    activation = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(activation.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activation[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Overlay heatmap on image
    img = np.array(image_pil.resize((224, 224)))[:, :, ::-1]
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Convert back to RGB for Streamlit
    result = Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))

    return result
