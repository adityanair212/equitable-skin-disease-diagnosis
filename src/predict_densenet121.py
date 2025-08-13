import torch
import numpy as np
from torchvision.models import densenet121, DenseNet121_Weights
from evaluate_single_model import predict_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = torch.nn.Linear(model.classifier.in_features, 5)
model.load_state_dict(torch.load("best_densenet121.pth", map_location=device))
model.to(device)

densenet_softmax = predict_model(model, "DenseNet121")
np.save("densenet_softmax.npy", densenet_softmax)
