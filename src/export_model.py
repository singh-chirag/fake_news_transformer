# scripts/export_model.py

import torch
from src.model import FakeNewsClassifier

checkpoint = torch.load("models/checkpoint.pt", map_location="cpu")

model = FakeNewsClassifier()
model.load_state_dict(checkpoint["model_state"])
model.eval()

torch.save(model.state_dict(), "models/model_prod.pt")
print("Production model saved as models/model_prod.pt")
