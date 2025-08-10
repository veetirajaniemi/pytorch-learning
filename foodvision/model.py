
import torch
import torchvision
from torch import nn

def create_effnetb2_model(num_classes: int = 101,
                          seed: int = 42):

  device = "cuda" if torch.cuda.is_available() else "cpu"

  weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  transforms = weights.transforms()
  model = torchvision.models.efficientnet_b2(weights=weights)

  for param in model.parameters():
    param.requires_grad = False

  torch.manual_seed(seed)
  model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1408,
              out_features=num_classes)).to(device)

  return model, transforms
