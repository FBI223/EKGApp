import torch
import coremltools as ct
from main_pytorch import UNet1D, WINDOW_SIZE
import numpy as np

def export_to_coreml():
    # === Wczytanie modelu PyTorch
    model = UNet1D(num_classes=4)  # 4 klasy: none, P, QRS, T
    model.load_state_dict(torch.load("unet.pt", map_location="cpu"))
    model.eval()

    # === Przykładowe wejście (1-lead, 2000 próbek, float32)
    example_input = torch.randn(1, 1, WINDOW_SIZE, dtype=torch.float32)

    # === Trace modelu (TorchScript)
    traced = torch.jit.trace(model, example_input)

    # === Konwersja do CoreML (.mlpackage)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",  # ML Program = nowszy runtime (iOS/macOS)
        inputs=[
            ct.TensorType(name="ecg", shape=(1, 1, WINDOW_SIZE), dtype=np.float32)
        ],
        compute_units=ct.ComputeUnit.ALL,  # opcjonalne: ALL / CPU_ONLY / CPU_AND_GPU
    )

    # === Zapis
    mlmodel.save("UnetModel.mlpackage")
    print("✅ Model zapisany jako UnetModel.mlpackage")

if __name__ == "__main__":
    export_to_coreml()
