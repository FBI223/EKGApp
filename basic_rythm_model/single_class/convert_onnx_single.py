
import torch
from main_single import SE_ResNet1D  # lub model_multi
import numpy as np

# Załaduj model
model = SE_ResNet1D(num_classes=9)
model.load_state_dict(torch.load("model_single.pt", map_location="cpu"))
model.eval()

# Przykładowe wejście
ecg_input = torch.randn(1, 1, 5000)
demo_input = torch.tensor([[60.0, 1.0]])

# Eksport do ONNX
torch.onnx.export(
    model,
    (ecg_input, demo_input),
    "model 3 no st  no georgia/model.onnx",
    input_names=["ecg", "demo"],
    output_names=["output"],
    dynamic_axes={"ecg": {0: "batch"}, "demo": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12
)

