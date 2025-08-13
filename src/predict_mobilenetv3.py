import torch
import numpy as np
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from evaluate_single_model import predict_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 5)
model.load_state_dict(torch.load("best_mobilenet_model.pth", map_location=device))
model.to(device)

mobilenet_softmax = predict_model(model, "MobileNetV3")
np.save("mobilenet_softmax.npy", mobilenet_softmax)
