import torch
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from evaluate_single_model import predict_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)
model.load_state_dict(torch.load("best_efficientnet_model.pth", map_location=device))
model.to(device)

effnet_softmax = predict_model(model, "EfficientNetB0")
np.save("effnet_softmax.npy", effnet_softmax)
