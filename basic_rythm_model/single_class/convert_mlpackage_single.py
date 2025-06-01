import torch
from main_single import SE_ResNet1D  # lub importuj klasę tak samo jak dla multi
import numpy as np
import coremltools as ct

# === Parametry modelu ===
NUM_CLASSES = 8 # Jedna etykieta z  9 możliwych

# === Wczytanie modelu ===
model = SE_ResNet1D(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("model_single.pt", map_location="cpu"))
model.eval()

# === Przykładowe dane wejściowe ===
example_input = torch.randn(1, 1, 5000)
example_demo = torch.tensor([[60.0, 1.0]])  # np. wiek i płeć

# === Trace modelu ===
traced = torch.jit.trace(model, (example_input, example_demo))

# === Konwersja do Core ML ===
mlmodel = ct.convert(
    traced,
    compute_precision=ct.precision.FLOAT16,
    inputs=[
        ct.TensorType(name="ecg", shape=(1, 1, 5000), dtype=np.float32),
        ct.TensorType(name="demo", shape=(1, 2), dtype=np.float32),
    ],
    convert_to="mlprogram"
)

# === Zapis ===
mlmodel.save("SE_ResNet1D_Single.mlpackage")
