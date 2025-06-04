import torch
import coremltools as ct
from model import ECGResNet34  # załaduj klasę modelu
import numpy as np

model = ECGResNet34(num_classes=9)
model.load_state_dict(torch.load("model_multi.pt", map_location="cpu"))
model.eval()

example_input = torch.randn(1, 12, 5000)  # 12-lead ECG, 10s @ 500Hz
traced = torch.jit.trace(model, example_input)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=example_input.shape)],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS15,
)

mlmodel.save("ECGResNet34.mlmodel")
print("✅ Zapisano ECGResNet34.mlmodel")
